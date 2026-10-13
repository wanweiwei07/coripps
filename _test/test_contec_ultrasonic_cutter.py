"""Manual test: ultrasonic cutter relay on CONTEC DIO-1616LN-USB (OUT0.2).

Waits for Enter, runs the cutter for ON_TIME seconds, then stops it and prints
the read-back state. Ctrl+C at any point switches everything off.

    py -3.12 coripps/_test/test_contec_ultrasonic_cutter.py
"""

import time

from coripps.drivers.contec_dio import ContecDIO

ON_TIME = 10.0

if __name__ == '__main__':
    with ContecDIO('DIO000') as dio:
        print('initial:', dio.state())
        input('press Enter -> ultrasonic_cutter ON for %.1f s ' % ON_TIME)
        dio.ultrasonic_cutter(True)
        print('  state:', dio.state())
        time.sleep(ON_TIME)
        dio.ultrasonic_cutter(False)
        st = dio.state()
        print('  state:', st)
        assert st['ultrasonic_cutter'] is False, st
        print('done')
