"""``main`` in simulation only: tube after tube, both robots, the gripper and
the capper, shown in the viewer -- no hardware touched at all.

Every cycle is ``single_routine.simulate`` -- the same steps, handshakes and
plans as on the cell, every move planned but only recorded -- from both
robots at home (``home_picker`` / ``home_sampler``), the next cycle from
where the last one ended. As in ``single_routine``'s simulation, the
sampler's stereo pair is replaced by the raw pairs the last real run saved
(``vision/sampler_stereo/observe/``, SAM 3 run on them, so the same leaf
every cycle; a relook that misses on them keeps the leaf as looked: the
pairs are not from the poses planned here). The tube is drawn too
(every tube the picker can take, in the rack at the start; the handled
one by ``tube_track``: where the steps put it -- not simulated, so pushed
and popped by rule). and the picker's look at the cap by each tube's long tab, random
(``--seed``). No waits (settling, sucking, blowing: real time only) and SAM 3
once per replayed picture, so a cycle plans in a few seconds. The viewer loops all the cycles; the gripper
opens and closes and the capper's finger turns as the steps say. A cycle
that does not plan ends the run there (the ones before it are shown).

    py -3.12 coripps/main_sim.py                 # tube 0, one cycle
    py -3.12 coripps/main_sim.py --tube 3 --n 5
    py -3.12 coripps/main_sim.py --n 100 --speed 8   # all, fast
    py -3.12 coripps/main_sim.py --n 100 --speed 4 --video   # to documents/videos/
    py -3.12 coripps/main_sim.py --n 100 --speed 60 --slow-ends --video
"""

import os
import time
import types
import hashlib
import argparse
import threading

import numpy as np

import coripps.picker_motion as pkm
import coripps.single_routine as sr
import coripps.tool.home_sampler as hs
import coripps.tool.tube.layout_cap as cap
import coripps.tool.tube.layout_pick as layout
import coripps.end_effectors.xh430_capper.xh430_capper as xcm
import one.utils.math as oum
import coripps.tool.tube.poses_io_pick as ppio
import coripps.tool.tube.tcp_moves as otm
from coripps.robots.mobisys import mobisys
import coripps.tool.leaf.suc_cut_store as scs
from coripps.tool.leaf.suc_cut_store import LeafSession
from coripps.tool.tube.pick_sequence_single import PickSession, HeldTubeClear
from coripps.vision.sampler_stereo import leaf_pipeline as lp
from coripps.vision.sampler_stereo import leaves as lv
import one.motion.core.planning_context as omppc


# rendered videos (not in git: .gitignore)
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'documents', 'videos')


class _NoReader:
    """No network: ``home_sampler._build`` would read the picker's joints."""

    def __init__(self, *args, **kwargs):
        pass

    def connect(self):
        raise OSError('simulation: no robot read')


def sim_picker(home_qs):
    """A ``PickSession`` without its arm, gripper or camera: the planning
    state only, standing at ``home_qs``."""
    s = object.__new__(PickSession)
    _, s.picker, sampler = mobisys()
    s.pln_ctx, _ = pkm.build_planning_context(s.picker, sampler)
    transit = pkm.build_transit_context(s.picker, sampler)
    s._transit = {False: transit, True: omppc.PlanningContext(
        collider=transit.collider, cd_step_size=transit.cd_step_size,
        constraints=(HeldTubeClear(
            s.picker, s.picker.gripper.tcp('grasp_center')),))}
    s.grip_current, s.grip, s.quit = None, None, False
    s.points, s.look_qs_all, _ = ppio.load_checked(ndof=s.picker.ndof,
                                                   layout=layout)
    s.moves = otm.TcpMoves(s)
    s.seg = s.cobotta = s.gripper = None
    at = {'q': np.asarray(home_qs, np.float32)}
    s.real_qs = lambda: s.pln_ctx.enforce_bounds(at['q'].copy())
    s.at = at
    return s


# no waits in the simulation: settling, sucking, blowing ... are real time
# only (a retry loop still gets a breath)
_SIM_TIME = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time)
                                     if not k.startswith('_')})
_SIM_TIME.sleep = lambda s: time.sleep(min(s, 0.02))
scs.time = _SIM_TIME
sr.time = _SIM_TIME


class _CachedSegmenter:
    """SAM 3, each (picture, prompt) segmented once: the replayed pairs are
    the same every cycle."""

    def __init__(self, segmenter):
        self.seg, self.cache = segmenter, {}

    def masks(self, image, *args, **kwargs):
        key = (hashlib.md5(image.tobytes()).hexdigest(), image.shape,
               repr(args), repr(sorted(kwargs.items())))
        if key not in self.cache:
            self.cache[key] = self.seg.masks(image, *args, **kwargs)
        return self.cache[key]

    def __getattr__(self, name):
        return getattr(self.seg, name)


def sim_sampler(home_qs):
    """A ``LeafSession`` without its arm, cutter or cameras (SAM 3 loaded:
    the saved pairs are segmented as live ones), standing at ``home_qs``."""
    s = object.__new__(LeafSession)
    s.tf_flange_cam = lp.load_handeye()
    print('loading SAM 3 ...')
    s.segmenter = _CachedSegmenter(lv.Segmenter())
    hs.Cobotta = _NoReader
    _, _, s.sampler, s.pln_ctx = hs._build()
    s.cobotta = s.cutter = s.eye = None
    s.leaf = s.plant = None
    s.leaves, s.tried, s.lifted, s.target = [], [], True, None  # as __init__
    at = {'q': np.asarray(home_qs, np.float32)}
    s.qs_now = lambda: s.pln_ctx.enforce_bounds(at['q'].copy())
    s.at = at
    return s


_relook = lp.relook


def _relook_or_keep(eye, sampler, qs, tf_flange_cam, segmenter, leaf,
                    **kwargs):
    """``lp.relook``, the leaf kept as looked when it misses: the replayed
    pairs (``observe/``, 2026-10-30) were taken from that run's poses and
    calibration, not from the ones planned here, so the relook may see the
    leaf moved -- a simulation shows the moves, not the vision."""
    new = _relook(eye, sampler, qs, tf_flange_cam, segmenter, leaf, **kwargs)
    if new is None:
        print('  simulation: the relook missed on the replayed pair -- the '
              'leaf kept as looked')
        return leaf
    return new


lp.relook = _relook_or_keep


def standing_tf(top, rotmat):
    """A tube standing upright, its cap's top at ``top``, turned ``rotmat``
    (its model's +x is its cap's long tab)."""
    return oum.tf_from_pos_rotmat(pos=np.asarray(top) + ppio.TUBE_POS_OFFSET,
                                  rotmat=rotmat)


def tab_rotmat(tab_deg):
    """An upright tube's turn with its cap's long tab at ``tab_deg``."""
    return oum.rotmat_from_axangle(np.array([0.0, 0.0, 1.0]),
                                   np.radians(tab_deg))


def tube_track(frames, point, tab_deg):
    """The handled tube's pose (4x4, world) at every
    ``single_routine.timeline`` frame of one cycle, from the steps' marks in
    them: in the rack at ``point`` (its cap's top, its long tab at
    ``tab_deg``, as the simulation's look takes it) until the
    grip; then held as gripped (fixed in the hand) until let go; standing in
    the store as let go -- pushed down by the closed hand over it, to home
    at most (the capper's 'tube' tcp) --; popped up (to the re-pick's cap
    height) once the capper is back home from its stopper; held again from
    the re-pick; gone at the discard (None)."""
    _, fk_picker, _ = mobisys()
    gc = fk_picker.gripper.tcp('grasp_center')
    home_z = float(cap.TUBE_TCP[2])
    popped_z = float(cap.REPICK_POS[2] - cap.REPICK_DESCEND)
    out, st, seen_stopper = [], 'rack', False
    tf = standing_tf(point, tab_rotmat(tab_deg))
    rel, top = None, None
    for p_qs, _, c_q, verb in frames:
        fk_picker.fk(qs=p_qs)
        g_tf = np.asarray(gc.tf, dtype=np.float64)
        if st in ('rack', 'popped') and verb == 'grip':
            rel = np.linalg.inv(g_tf) @ tf          # held as it is gripped
            st = 'held' if st == 'rack' else 'held2'
        if st in ('held', 'held2'):
            tf = g_tf @ rel
            if verb == 'open':
                if st == 'held2':
                    st, tf = 'gone', None
                else:
                    st = 'placed'
                    top = tf[:3, 3] - ppio.TUBE_POS_OFFSET
        elif st == 'placed':
            tip = g_tf[2, 3] - cap.FINGERTIP_BELOW_GC
            if (verb == 'close' and np.hypot(*(g_tf[:2, 3] - top[:2])) < 0.02
                    and tip < top[2]):
                top = np.array([top[0], top[1], max(home_z, tip)])
                tf = standing_tf(top, tf[:3, :3])
            if c_q == xcm.MODES['stopper']:
                seen_stopper = True
            elif seen_stopper and c_q == xcm.MODES['home']:
                st = 'popped'
                top = np.array([top[0], top[1], popped_z])
                tf = standing_tf(top, tf[:3, :3])
        out.append(tf)
    return out


def main(tube=0, n=1, speed=1, seed=0, video=None, slow_ends=None):
    picker = sim_picker(np.deg2rad(np.asarray(sr.HOME_QS_DEG, np.float32)))
    sampler = sim_sampler(np.deg2rad(np.asarray(sr.SAMPLER_HOME_QS_DEG,
                                                np.float32)))
    phases, tubes, done = [], [], set()
    starts = []                 # each cycle's first timeline frame
    # every tube the picker can take, standing in the rack at the start
    # each tube's cap long tab: random (``seed``), as it may sit in the rack
    tabs = np.random.default_rng(seed).uniform(0.0, 360.0, len(picker.points))
    rack = [standing_tf(pt, tab_rotmat(tab))
            for pt, tab in zip(picker.points, tabs)]
    first = threading.Event()

    def plan_all():
        """The cycles one after another, each appended as it is planned (the
        viewer plays what is there)."""
        try:
            for k in range(tube, min(tube + n, len(picker.points))):
                print(f'\n==== cycle: tube {k} ====')
                try:
                    picker.select(k)
                except RuntimeError as exc:
                    print(f'tube {k}: {exc} -- stopping')
                    break
                plan = sr.plan_picker(picker)
                if plan is None:
                    print(f'tube {k}: not runnable -- stopping')
                    break
                sr.SIM_TAB_DEG = float(tabs[k])     # what the look will see
                cycle, ok = sr.simulate(picker, sampler, plan)
                track = tube_track(sr.timeline(cycle)[:-sr.HOLD_FRAMES],
                                   picker.points[k], float(tabs[k]))
                others = [rack[j] for j in range(len(rack))
                          if j != k and j not in done]
                done.add(k)
                # the tubes first: the viewer's frames follow the phases
                if tubes:
                    del tubes[-sr.HOLD_FRAMES:]
                tubes.extend(others + ([] if t is None else [t])
                             for t in track)
                tubes.extend([tubes[-1]] * sr.HOLD_FRAMES)
                starts.append(len(tubes) - sr.HOLD_FRAMES - len(track))
                phases.extend(cycle)
                first.set()
                print(f'cycles planned: {len(done)}', flush=True)
                if not ok:
                    print(f'tube {k}: the simulation FAILED there -- shown '
                          'up to it')
                    break
                # the next cycle from where this one ended
                picker.at['q'] = np.asarray(cycle[-1][0][-1], np.float32)
                sampler.at['q'] = np.asarray(cycle[-1][1][-1], np.float32)
        finally:
            first.set()
            print('planning done', flush=True)

    if video is not None:               # every cycle first, then render
        plan_all()
        if not phases:
            return False
        if slow_ends and len(starts) > 1:
            # the first and the last cycle ``slow_ends`` times faster, the
            # rest ``speed`` times
            first_end, last_start = starts[1], starts[-1]
            pace = lambda i: (slow_ends if i < first_end or i >= last_start
                              else speed)
        else:
            pace = speed
        sr.record(phases, video, tube_tfs=tubes, speed=pace)
        return True
    worker = threading.Thread(target=plan_all, name='plan', daemon=True)
    worker.start()
    first.wait()                        # the viewer opens on the first cycle
    if not phases:
        return False
    print('viewer: the cycles play as they are planned')
    sr.show(phases, False, None, tube_tfs=tubes, speed=speed,
            growing=worker.is_alive)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tube', type=int, default=0,
                        help='the first precomputed pick index (default 0)')
    parser.add_argument('--n', type=int, default=1,
                        help='how many cycles (default 1)')
    parser.add_argument('--speed', type=int, default=1,
                        help='replay this many times faster (default 1)')
    parser.add_argument('--seed', type=int, default=0,
                        help="the tubes' random long-tab angles (default 0)")
    parser.add_argument('--video', nargs='?', const='', default=None,
                        help='render to a video in coripps/documents/videos instead of '
                             'showing (a name, default main_sim_<tube>_<n>'
                             '.mp4); --speed sets its replay speed')
    parser.add_argument('--slow-ends', type=int, nargs='?', const=3,
                        default=None,
                        help='with --video: the first and the last cycle at '
                             'this speed (default 3), --speed for the rest')
    args = parser.parse_args()
    video = None
    if args.video is not None:
        name = args.video or f'main_sim_{args.tube}_{args.n}.mp4'
        video = os.path.join(VIDEO_DIR, name)
    main(args.tube, args.n, args.speed, args.seed, video, args.slow_ends)
