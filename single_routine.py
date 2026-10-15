"""One tube and one plant, both robots at once: the picker brings a tube to
the capper and opens it, the sampler takes a leaf and drops it into the
tube, the capper closes it, the picker pops the tube out and discards it.

    picker (+ gripper, capper)              sampler (+ cutter, D405)
    --------------------------              ------------------------
    home (the start)                        home (the start)
    ---- both there: go on ------------------------------------------------
    pick the tube, place it in the          observe, look, over the plant,
    capper, push it home                    look again, onto the leaf, suck,
      (pick_cap_store steps 1-14)           cut, lift
    back home                                 (suc_cut_store steps 1-12)
    capper: open the cap (the arm away)
    ---- CAP_OPEN ---------------------->   wait for CAP_OPEN
                                            over the tube, into it, blow,
                                            back up out of it
    wait for TUBE_FREE   <--------------    ---- TUBE_FREE
    capper: close the cap, stopper,         back home
    meanwhile the arm to the pop position
      as soon as a free path is there (the
      sampler still over the capper blocks
      it: re-planned every RETRY_PERIOD);
    pop the tube, capper home, re-pick it,
    discard it, home  (pick_cap_store
    steps 15-26)

Each robot starts at, retreats to and ends at its home (``home_picker`` /
``home_sampler``): the picker's grasp_center at (100, -100, 100) mm, the
sampler's cutter tcp at (100, -50, 100) mm, each in its own base frame,
pointing down.

FIRST A SIMULATION of the whole routine, from both robots' real joints:
the same steps and handshakes, every move planned but only recorded, the
grippers / capper / cutter only logged, the sampler's camera replaced by the
frame of its last observation (``vision/d405/observe/``; the look again
keeps the first estimate). The viewer loops it -- 0: both to their start,
A: both until CAP_OPEN, B: the sampler storing the leaf, C: the sampler
retreating while the picker pops and discards. A step that fails in the
simulation is printed and nothing can be run. 'g' in the viewer window then runs it on the real cell
and the viewer follows the real arms.

On the real cell each robot runs in its own thread with its own
connections. Before every step a thread re-reads the OTHER robot's joints
and rebuilds its collision model with them (``refresh_other``), so a move is
planned / checked against where the other arm really is at that moment --
but the other arm may still move during it: the cell's zones and the three
handshakes above are what keep them apart.

'q' in the viewer (or closing it) stops both before their next step (a
running move finishes); a failure in one stops the other too. On the
sampler's end its cutter board is closed: the ultrasonic and the vacuum go
off (a held leaf drops).

    py -3.12 coripps/single_routine.py
    py -3.12 coripps/single_routine.py --tube 3
"""

import sys
import time
import argparse
import threading

import numpy as np

from one.drivers.cobotta import BCapError, Cobotta

import coripps.tool.tube.pick_cap_store as pcs
from coripps.end_effectors.xh430_capper.exec import (XH430CapperExec,
                                                     DynamixelError)
from coripps.robots.leaf_sampler import LeafSampler
from coripps.robots.tube_picker import TubePicker
from coripps.tool.home_sampler import _HOME_QS_DEG as SAMPLER_HOME_QS_DEG
from coripps.tool.leaf.suc_cut_store import LeafSession
from coripps.tool.tube.home_picker import HOME_QS_DEG
from coripps.tool.tube.pick_sequence_single import PickSession
from coripps.vision.d405 import leaf_pipeline as lp


RETREAT_SPEED = 100.0                       # %, both robots going home
WAIT_POLL = 0.1                             # s
RETRY_PERIOD = 0.5                          # s, re-planning while blocked
PLAY_INTERVAL = 1 / 30.0                    # s, the animation
HOLD_FRAMES = 30                            # held at the end of the loop
FOLLOW_INTERVAL = 0.1                       # s, the real arms on screen
_SAMPLER_HOST = '192.168.0.101'
_PICKER_HOST = '192.168.0.102'


class Shared:
    """The handshakes between the two threads, the stop flag, where the
    other robot's joints come from, and (in the simulation) the phase
    marks."""

    def __init__(self, other_qs=None, mark=None, stop=None):
        self.picker_ready = threading.Event()   # at its start pose
        self.sampler_ready = threading.Event()  # at its start pose
        self.sampler_done = threading.Event()   # at its retreat, finished
        self.cap_open = threading.Event()       # picker -> sampler
        self.tube_free = threading.Event()      # sampler -> picker
        # ``stop``: one run's own, or one shared by several runs
        self.stop = threading.Event() if stop is None else stop
        self.other_qs = other_qs or read_qs     # (host, ndof) -> joints
        self.mark = mark or (lambda name: None)

    def wait(self, event, who, what):
        """Wait for ``event``; False when the run is stopped meanwhile."""
        print(f'{who}: waiting for {what}')
        while not event.wait(WAIT_POLL):
            if self.stop.is_set():
                return False
        print(f'{who}: {what} -- going on')
        return True


def read_qs(host, ndof):
    with Cobotta(host) as reader:
        return reader.joint_angles(ndof=ndof)


def refresh_other(shared, pln_ctx, other_cls, host):
    """Put the other robot at its current joints in this collision model."""
    other = next(m for m in pln_ctx.collider.scene.mecbas
                 if isinstance(m, other_cls))
    other.fk(qs=shared.other_qs(host, other.ndof))
    pln_ctx.collider.refresh()
    pln_ctx.clear_cache()


# -- the picker ---------------------------------------------------------------
def picker_home(session):
    """Step: the picker home (``home_picker.HOME_QS_DEG``)."""
    goal = session.pln_ctx.enforce_bounds(np.deg2rad(np.asarray(
        HOME_QS_DEG, np.float32)))
    return session.transfer_to('home', goal, speed=RETREAT_SPEED)


def to_pop_when_free(shared, session, plan):
    """The arm to the pop position as soon as there is a free path: the
    sampler, just out of the tube, may still be over the capper -- planned
    again (the sampler re-read) every RETRY_PERIOD until it goes, given up
    when the sampler has finished and there is still none."""
    print('picker: to the pop position (once the sampler is out of the way)')
    while True:
        if shared.stop.is_set():
            return False
        finished = shared.sampler_done.is_set()
        refresh_other(shared, session.pln_ctx, LeafSampler, _SAMPLER_HOST)
        if session.transfer_to('the pop position', plan['pop'],
                               speed=pcs.FREE_SPEED):
            return True
        if finished:
            print('picker: still no path to the pop position with the '
                  'sampler gone -- stopping')
            return False
        print(f'picker: blocked by the sampler, again in {RETRY_PERIOD} s')
        time.sleep(RETRY_PERIOD)


def picker_flow(shared, session, capper, plan):
    """The picker's thread: ``(label, step)`` lists between the
    handshakes, every step after re-reading the sampler."""
    steps = pcs.tube_steps(session, capper, plan)
    before = steps[:14]                 # pick, place, push (steps 1-14)
    after = steps[15:]                  # press the lever ... home (16-26)

    def run(label, step):
        if shared.stop.is_set():
            return False
        print(f'picker: {label}')
        refresh_other(shared, session.pln_ctx, LeafSampler, _SAMPLER_HOST)
        if not step():
            print(f'picker: "{label}" failed')
            return False
        return True

    def open_cap():
        capper.set_opening('open')
        print(f'  capper open at {capper.servo_deg():6.2f} deg')
        return True

    def close_cap(result):
        try:
            capper.set_opening('close')
            result['close'] = capper.servo_deg()
            capper.set_opening('stopper')
            result['stopper'] = capper.servo_deg()
        except DynamixelError as exc:
            result['error'] = exc

    try:
        if not run('to the start: home', lambda: picker_home(session)):
            return False
        shared.mark('picker_0')
        shared.picker_ready.set()
        if not shared.wait(shared.sampler_ready, 'picker',
                           'the sampler at its start'):
            return False
        for label, step in before + [
                ('back home, away from the capper',
                 lambda: picker_home(session)),
                ('capper: open the cap (the arm away)', open_cap)]:
            if not run(label, step):
                return False
        shared.mark('picker_a')
        shared.cap_open.set()
        if not shared.wait(shared.tube_free, 'picker', 'the tube free'):
            return False
        result = {}
        worker = threading.Thread(target=close_cap, args=(result,),
                                  name='capper')
        worker.start()                  # the cap closes while the arm ...
        try:
            shared.mark('picker_d')     # ... goes to the pop position
            ok = to_pop_when_free(shared, session, plan)
        finally:
            worker.join()
        if 'error' in result:
            print(f"  capper failed: {result['error']}")
            return False
        print(f"  capper pressed shut at {result['close']:6.2f} deg, at the "
              f"stopper {result['stopper']:6.2f} deg")
        if not ok:
            return False
        for label, step in after:
            if not run(label, step):
                return False
        print('picker: done -- the tube is discarded')
        return True
    except (OSError, BCapError, DynamixelError, RuntimeError) as exc:
        print(f'picker: {exc} -- stopping')
        return False


# -- the sampler --------------------------------------------------------------
def sampler_home(session, label='back home'):
    """Step maker: the sampler home (``home_sampler._HOME_QS_DEG``)."""
    start = session.qs_now()
    goal = session.pln_ctx.enforce_bounds(np.deg2rad(np.asarray(
        SAMPLER_HOME_QS_DEG, np.float32)))
    path = (lp.plan(session.pln_ctx, start, goal)
            if session.pln_ctx.is_state_valid(goal) else None)
    if path is None:
        raise RuntimeError('sampler home: in collision / no path')
    return (label, lambda: session.run(path, RETREAT_SPEED, 'retreat'))


def sampler_flow(shared, session):
    """The sampler's thread."""
    makers = session.steps()
    cut = makers.index(session.s_to_capper)
    before, store = makers[:cut], makers[cut:]  # up to the lift / the store

    def run(make):
        if shared.stop.is_set():
            return False
        refresh_other(shared, session.pln_ctx, TubePicker, _PICKER_HOST)
        label, step = make()            # planned against the fresh model
        print(f'sampler: {label}')
        step()
        return True

    try:
        if not run(lambda: sampler_home(session, 'to the start: home')):
            return False
        shared.mark('sampler_0')
        shared.sampler_ready.set()
        if not shared.wait(shared.picker_ready, 'sampler',
                           'the picker at its start'):
            return False
        for make in before:
            if not run(make):
                return False
        if not shared.wait(shared.cap_open, 'sampler',
                           'the cap open and the picker away'):
            return False
        shared.mark('sampler_b')
        for make in store:
            if not run(make):
                return False
        shared.mark('sampler_c')
        shared.tube_free.set()          # out of the tube: the cap may close
        if not run(lambda: sampler_home(session)):
            return False
        shared.sampler_done.set()
        print('sampler: done -- the leaf is in the tube')
        return True
    except (OSError, BCapError, RuntimeError) as exc:
        print(f'sampler: {exc} -- stopping')
        return False


def run_both(shared, picker, capper, plan, sampler):
    """Start both threads; ``{'picker': bool, 'sampler': bool}`` fills in
    as they end, a failure of one sets ``shared.stop`` for the other."""
    results = {}

    def go(name, flow, *args):
        results[name] = False
        try:
            results[name] = flow(shared, *args)
        except Exception as exc:        # never leave the other one waiting
            print(f'{name}: {exc!r} -- stopping')
        finally:
            if results[name] is not True:
                shared.stop.set()

    threads = [threading.Thread(target=go, args=('picker', picker_flow,
                                                 picker, capper, plan),
                                name='picker', daemon=True),
               threading.Thread(target=go, args=('sampler', sampler_flow,
                                                 sampler),
                                name='sampler', daemon=True)]
    for t in threads:
        t.start()
    return threads, results


# -- the simulation -------------------------------------------------------------
class _SimCapper:
    def __init__(self, log):
        self.log = log

    def set_opening(self, mode):
        self.log.append(f'capper {mode}')

    def servo_deg(self):
        return float('nan')


class _SimCutter:
    def __init__(self, log):
        self.log = log

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.log.append(
            f'cutter {name}{args if args else ""}')


def simulate(picker, sampler, plan):
    """The whole routine in simulation (module doc). ``(phases, ok)``:
    phases 0, A-D as ``(picker frames, sampler frames)``. The sessions are put
    back as they were afterwards."""
    sim = {'p': picker.real_qs(), 's': sampler.qs_now()}
    frames = {'p': [np.asarray(sim['p'], np.float32)],
              's': [np.asarray(sim['s'], np.float32)]}
    marks, log = {}, []
    color, depth, K, obs_qs = lp.load_observation()

    def p_play(name, waypoints, speed=None):
        frames['p'] += lp.densify(waypoints)[1:]
        sim['p'] = np.asarray(waypoints[-1], np.float32)
        return True

    def s_run(path, speed, what):
        frames['s'] += lp.densify(path)[1:]
        sim['s'] = np.asarray(path[-1], np.float32)

    def s_observe():                    # to where the saved frame was taken
        path = lp.plan(sampler.pln_ctx, sampler.qs_now(),
                       sampler.pln_ctx.enforce_bounds(obs_qs))
        if path is None:
            raise RuntimeError('observe: no path to the saved frame\'s pose')
        return ('to the observation pose (the saved frame\'s)',
                lambda: s_run(path, None, 'observe'))

    patches = [(picker, 'real_qs', lambda: picker.pln_ctx.enforce_bounds(
                    sim['p'].copy())),
               (picker, 'play', p_play),
               (picker, 'work_gripper',
                lambda verb: log.append(f'gripper {verb}') or True),
               (sampler, 'qs_now', lambda: sampler.pln_ctx.enforce_bounds(
                   sim['s'].copy())),
               (sampler, 'run', s_run),
               (sampler, 'frame', lambda: (color, depth, K)),
               (sampler, 's_observe', s_observe),
               (sampler, 'cutter', _SimCutter(log)),
               (lp, 'relook', lambda job, old: (old, None))]
    saved = [(obj, name, obj.__dict__.get(name)) for obj, name, _ in patches]
    for obj, name, value in patches:
        setattr(obj, name, value)
    shared = Shared(
        other_qs=lambda host, ndof: sim['s'] if host == _SAMPLER_HOST
        else sim['p'],
        mark=lambda name: marks.__setitem__(
            name, len(frames['p' if name.startswith('picker') else 's'])))
    try:
        print('\n==== simulation ====')
        threads, results = run_both(shared, picker, _SimCapper(log), plan,
                                    sampler)
        for t in threads:
            t.join()
    finally:
        for obj, name, value in saved:
            if value is None:
                delattr(obj, name)
            else:
                setattr(obj, name, value)
    ok = results.get('picker') is True and results.get('sampler') is True
    print('simulation: ' + ('every step plans' if ok else
                            'FAILED -- nothing can be run'))
    print('  ' + ', '.join(log))
    fp, fs = frames['p'], frames['s']
    p0, s0 = marks.get('picker_0', len(fp)), marks.get('sampler_0', len(fs))
    pa, pd = marks.get('picker_a', len(fp)), marks.get('picker_d', len(fp))
    sb = marks.get('sampler_b', len(fs))
    sc = marks.get('sampler_c', len(fs))
    phases = [(fp[:p0], fs[:s0]),               # 0: to the start
              (fp[p0 - 1:pa], fs[s0 - 1:sb]),   # A
              ([fp[pa - 1]], fs[sb - 1:sc]),    # B
              (fp[pd - 1:] if pd < len(fp) else [fp[-1]], fs[sc - 1:])]  # C
    return phases, ok


# -- the viewer -------------------------------------------------------------
def timeline(phases):
    """Per animation frame ``(picker qs, sampler qs)``: the phases one
    after another, both robots in step inside a phase."""
    out = []
    for fp, fs in phases:
        n = max(len(fp), len(fs))
        out += [(fp[min(k, len(fp) - 1)], fs[min(k, len(fs) - 1)])
                for k in range(n)]
    return out + [out[-1]] * HOLD_FRAMES


def show(phases, ok, real):
    """Loop the simulation; 'g' starts ``real()`` (when ``ok``) and the
    viewer follows the real arms; 'q' or closing stops it."""
    from pyglet.window import key
    import one.viewer.world as ovw
    from coripps.robots.mobisys import mobisys

    frames = timeline(phases)
    base = ovw.World(cam_pos=(1.1, 0.5, 0.8),
                     cam_lookat_pos=(0.30, -0.10, 0.10))
    mobisys_base, picker, sampler = mobisys()
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    state = {'i': 0, 'running': None}
    print("\nviewer: the simulation loops -- " +
          ("'g' in the window runs it on the real cell, " if ok else
           'it FAILED, nothing can be run; ') + "'q' / closing stops")

    def follow():
        for robot, host in ((picker, _PICKER_HOST), (sampler, _SAMPLER_HOST)):
            try:
                robot.fk(qs=read_qs(host, robot.ndof))
            except (OSError, BCapError):
                pass

    def tick(dt):
        im = base.input_manager
        go = im.is_key_pressed_edge(key.G)
        if im.is_key_pressed_edge(key.Q) and state['running'] is not None:
            print('q: stopping both before their next step')
            state['running'].stop.set()
        if state['running'] is None:
            if go and ok:
                state['running'] = real()
                base.schedule_interval(lambda dt: follow(),
                                       interval=FOLLOW_INTERVAL)
                return
            p_qs, s_qs = frames[state['i']]
            picker.fk(qs=p_qs)
            sampler.fk(qs=s_qs)
            state['i'] = (state['i'] + 1) % len(frames)

    base.schedule_interval(tick, interval=PLAY_INTERVAL)
    base.run()
    return state['running']


def plan_picker(picker):
    """The tube's run (``pick_cap_store.plan_tube``), its fixed poses
    checked with the sampler at its HOME pose, not where it stands now: it
    leaves for the plant first, and every step re-checks against the real
    sampler anyway. None (said why) when not runnable."""
    cap_poses = pcs.load_cap_poses(picker)
    if cap_poses is None:
        return None
    shared = Shared(other_qs=lambda host, ndof: np.deg2rad(np.asarray(
        SAMPLER_HOME_QS_DEG, np.float32)))
    refresh_other(shared, picker.pln_ctx, LeafSampler, _SAMPLER_HOST)
    return pcs.plan_tube(picker, cap_poses, None)


def main(tube=pcs.TUBE_INDEX):
    try:
        picker = PickSession(tube)
    except RuntimeError as exc:
        print(exc)
        return False
    plan = plan_picker(picker)
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
    pcs.print_plan(plan)
    run = {}
    try:
        phases, ok = simulate(picker, sampler, plan)

        def real():
            print('\n==== the real cell ====')
            shared = Shared()
            run['threads'], run['results'] = run_both(shared, picker, capper,
                                                      plan, sampler)
            return shared

        shared = show(phases, ok, real)
        if shared is None:
            return False
        shared.stop.set()               # the window is closed: stop both
        for t in run['threads']:
            t.join()
        results = run['results']
        done = results.get('picker') is True and results.get('sampler') is True
        print('\nall done' if done else '\nnot finished')
        return done
    finally:
        sampler.close()                 # ultrasonic / vacuum off, arm back
        capper.close()                  # the port only
        picker.release()                # arm back; the gripper port only


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tube', type=int, default=pcs.TUBE_INDEX,
                        help=f'precomputed pick index (default '
                             f'{pcs.TUBE_INDEX})')
    args = parser.parse_args()
    sys.exit(0 if main(tube=args.tube) else 1)
