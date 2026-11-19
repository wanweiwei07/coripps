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
``single_routine.py`` for one). A live view (``coripps.monitor``) shows
the cell as it is: both robots, the rack's tubes (the worked one red), the
sampler's last look -- its point cloud, and a picture window with the
leaves, the chosen one and every leaf's numbers. 'g' (in the viewer or the
terminal) runs all the cycles back to back; 'q' / Esc stops both robots
before their next step; closing the viewer stops them too.

    py -3.12 coripps/main.py                 # tube 0 on, until they run out
    py -3.12 coripps/main.py --tube 3 --n 5
"""

import sys
import argparse
import threading

try:
    import msvcrt                       # the terminal's keys (Windows)
except ImportError:
    msvcrt = None

from coripps import single_routine as sr
from coripps.end_effectors.xh430_capper.exec import (XH430CapperExec,
                                                     DynamixelError)
from coripps import monitor
from coripps.tool.keys import _QUIT_KEYS
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
            monitor.publish_tubes(tube, self.done)
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
            monitor.publish_tubes(None, self.done)
            print(f'cycle {k + 1}: tube {tube} done')
        print(f'\n{len(self.done)} of {len(self.tubes)} cycles done '
              f'(tubes {self.done})')

    def start(self):
        self.thread.start()
        return self


def watch(picker, capper, sampler, tubes):
    """The live view (``coripps.monitor``): the cell as it really is, the
    rack's tubes (the worked one red, the done ones gone), the sampler's
    last look -- its point cloud, its picture with the leaves and their
    numbers. 'g' (in the viewer or the terminal) runs the cycles; 'q' /
    Esc stops both robots before their next step; closing the window
    stops them too. The Runner, or None when never started."""
    from pyglet.window import key
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=sr.CAM_POS, cam_lookat_pos=sr.CAM_LOOKAT)
    stage = sr._Stage(base)
    monitor.publish_tubes(None)
    monitor.Live(base, stage, capper, picker.points).start()
    state = {'runner': None}
    print(f"\n{len(tubes)} cycles: tubes {tubes[0]}..{tubes[-1]} -- 'g' (in "
          "the viewer or here) runs them, 'q' stops, closing the window "
          'stops too')

    def tick(dt):
        im = base.input_manager
        typed = None
        if msvcrt is not None:
            while msvcrt.kbhit():
                typed = msvcrt.getwch().lower()
        go = im.is_key_pressed_edge(key.G) or typed == 'g'
        quit_ = im.is_key_pressed_edge(key.Q) or typed in _QUIT_KEYS
        runner = state['runner']
        if runner is None:
            if go:
                state['runner'] = Runner(picker, capper, sampler,
                                         tubes).start()
        elif quit_ and not runner.stop.is_set():
            print('q: stopping both robots before their next step')
            runner.stop.set()

    base.schedule_interval(tick, interval=sr.WAIT_POLL)
    base.run()
    return state['runner']


def main(first=0, n=None):
    try:
        picker = PickSession(first)
    except RuntimeError as exc:
        print(exc)
        return False
    last = len(picker.points) if n is None else min(first + n,
                                                    len(picker.points))
    tubes = list(range(first, last))
    if not tubes:
        print(f'no tubes from {first} on ({len(picker.points)} precomputed)')
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
        capper.disconnect()
        picker.release()
        return False
    try:
        runner = watch(picker, capper, sampler, tubes)
        if runner is None:
            return False
        runner.stop.set()               # the window closed: stop both
        runner.thread.join()
        return len(runner.done) == len(tubes)
    finally:
        sampler.close()                 # ultrasonic / vacuum off, arm back
        capper.disconnect()             # the port only
        picker.release()                # arm back; the gripper port only
        sr.close_readers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tube', type=int, default=0,
                        help='the first precomputed tube (default 0)')
    parser.add_argument('--n', type=int, default=None,
                        help='how many cycles (default: until the tubes '
                             'run out)')
    args = parser.parse_args()
    sys.exit(0 if main(first=args.tube, n=args.n) else 1)
