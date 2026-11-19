"""Switch the vacuum generator (CONTEC DIO-1616LN-USB) off: suck and blow.

Headless, nothing to press. The ultrasonic is left as it is.

    py -3.12 coripps/tool/contec/vacuum_off.py
"""

from coripps.drivers.contec_dio import ContecDIO


def vacuum_off():
    """Vacuum off; the board's outputs after it."""
    with ContecDIO('DIO000') as dio:
        dio.vacuum_generator('off')
        state = dio.state()
    print('state:', state)
    return state


if __name__ == '__main__':
    vacuum_off()
