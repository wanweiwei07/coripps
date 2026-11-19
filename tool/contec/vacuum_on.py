"""Switch the vacuum generator (CONTEC DIO-1616LN-USB) on -- suck -- and
keep it on until Enter (or Ctrl+C): closing the board switches every output
off, so it sucks only while this runs. The pressure sensor's seal is shown
as it changes.

    py -3.12 coripps/tool/contec/vacuum_on.py
"""

import threading
import time

from coripps.drivers.contec_dio import ContecDIO

SEAL_INPUT = 0                  # the pressure sensor's input (as the cutter's)


def vacuum_on():
    """Suck until Enter / Ctrl+C, then off."""
    with ContecDIO('DIO000') as dio:
        dio.vacuum_generator('suck')
        print('state:', dio.state(), "-- Enter (or Ctrl+C) to switch it off")
        done = threading.Event()
        threading.Thread(target=lambda: (input(), done.set()),
                         daemon=True).start()
        t0, last = time.time(), None
        try:
            while not done.is_set():
                sealed = dio.input(SEAL_INPUT)
                if sealed != last:
                    print(f'{time.time() - t0:6.2f} s  sealed: {sealed}')
                    last = sealed
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            dio.vacuum_generator('off')
            print('state:', dio.state())


if __name__ == '__main__':
    vacuum_on()
