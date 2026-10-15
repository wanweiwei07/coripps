"""Tube after tube, leaf after leaf: ``single_routine`` over and over.

Every cycle is one ``single_routine`` run -- both robots home, the picker
places the next tube and opens it, the sampler cuts a leaf and drops it in,
the cap is closed, the tube popped out and discarded, both robots home
again -- with the next precomputed tube (from ``--tube`` on) and the leaf
the sampler finds best this time (it looks at the plant afresh every
cycle: the leaves already cut are gone). ``--n`` cycles, or until the tubes
run out; a cycle that does not finish (no leaf left, no path, a failure,
'q') ends the run.

The hardware is connected and SAM 3 loaded once; no simulation (run
``single_routine.py`` for one). 'g' in the terminal runs all the cycles
back to back; 'q' / Esc / Ctrl+C stops both robots before their next
step.

    py -3.12 coripps/main.py                 # tube 0 on, until they run out
    py -3.12 coripps/main.py --tube 3 --n 5
"""

import sys
import time
import argparse
import threading

from coripps import single_routine as sr
from coripps.end_effectors.xh430_capper.exec import (XH430CapperExec,
                                                     DynamixelError)
from coripps.tool.keys import _quit_pressed, _wait_for_g
from coripps.tool.leaf.suc_cut_store import LeafSession
from coripps.tool.tube.pick_sequence_single import PickSession


class Runner:
    """The real cycles, in their own thread; ``stop`` stops them all."""

    def __init__(self, picker, capper, sampler, tubes):
        self.picker, self.capper, self.sampler = picker, capper, sampler
        self.tubes = tubes
        self.stop = threading.Event()
        self.done = []                  # the tubes whose cycle finished
        self.thread = threading.Thread(target=self.run, name='cycles',
                                       daemon=True)

    def run(self):
        for k, tube in enumerate(self.tubes):
            if self.stop.is_set():
                break
            print(f'\n==== cycle {k + 1}/{len(self.tubes)}: tube {tube} ====')
            try:
                self.picker.select(tube)
            except RuntimeError as exc:
                print(f'tube {tube}: {exc} -- stopping')
                break
            plan = sr.plan_picker(self.picker)
            if plan is None:
                print(f'tube {tube}: not runnable -- stopping')
                break
            shared = sr.Shared(stop=self.stop)
            threads, results = sr.run_both(shared, self.picker, self.capper,
                                           plan, self.sampler)
            for t in threads:
                t.join()
            if not (results.get('picker') is True
                    and results.get('sampler') is True):
                print(f'cycle {k + 1} (tube {tube}) did not finish -- '
                      'stopping')
                self.stop.set()
                break
            self.done.append(tube)
            print(f'cycle {k + 1}: tube {tube} done')
        print(f'\n{len(self.done)} of {len(self.tubes)} cycles done '
              f'(tubes {self.done})')

    def start(self):
        self.thread.start()
        return self


def main(first=0, n=None):
    try:
        picker = PickSession(first)
    except RuntimeError as exc:
        print(exc)
        return False
    last = len(picker.paths) if n is None else min(first + n,
                                                    len(picker.paths))
    tubes = list(range(first, last))
    if not tubes:
        print(f'no tubes from {first} on ({len(picker.paths)} precomputed)')
        picker.release()
        return False
    plan = sr.plan_picker(picker)
    if plan is None:
        picker.release()
        return False
    try:
        capper = XH430CapperExec().connect()
    except DynamixelError as exc:
        print(f'capper: {exc}')
        picker.release()
        return False
    try:
        sampler = LeafSession()
    except Exception as exc:
        print(f'sampler: cannot start: {exc}')
        capper.close()
        picker.release()
        return False
    try:
        if not _wait_for_g(f'{len(tubes)} cycles: tubes '
                           f'{tubes[0]}..{tubes[-1]}'):
            return False
        runner = Runner(picker, capper, sampler, tubes).start()
        while runner.thread.is_alive():
            if _quit_pressed() and not runner.stop.is_set():
                print('q: stopping both robots before their next step')
                runner.stop.set()
            time.sleep(sr.WAIT_POLL)
        return len(runner.done) == len(tubes)
    finally:
        sampler.close()                 # ultrasonic / vacuum off, arm back
        capper.close()                  # the port only
        picker.release()                # arm back; the gripper port only


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tube', type=int, default=0,
                        help='the first precomputed tube (default 0)')
    parser.add_argument('--n', type=int, default=None,
                        help='how many cycles (default: until the tubes '
                             'run out)')
    args = parser.parse_args()
    sys.exit(0 if main(first=args.tube, n=args.n) else 1)
