"""Manual test: vacuum generator on CONTEC DIO-1616LN-USB (OUT0.0 suck, OUT0.1 blow).

Steps through suck -> off -> blow -> off -> suck -> blow -> off, waiting for
Enter between steps and printing the read-back state. The last two steps
switch suck -> blow directly to check the interlock. Ctrl+C at any point
switches everything off.

    py -3.12 coripps/_test/test_contec_vacuum_generator.py
"""

from coripps.drivers.contec_dio import ContecDIO

STEPS = ['suck', 'off', 'blow', 'off', 'suck', 'blow', 'off']

if __name__ == '__main__':
    with ContecDIO('DIO000') as dio:
        print('initial:', dio.state())
        for mode in STEPS:
            input("press Enter -> vacuum_generator('%s') " % mode)
            dio.vacuum_generator(mode)
            st = dio.state()
            print('  state:', st)
            assert st['vacuum_generator'] == mode, st
        print('done')
