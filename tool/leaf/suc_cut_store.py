"""Suck, cut and store one leaf, step by step: find the plant and look
at it with the sampler's stereo cameras, line the cutter up with the
chosen leaf, take it with the nozzle, cut its petiole with the ultrasonic
blade, lift it, and put it into the tube in the capper.

    1   to the search pose: the stereo pair SEARCH_HEIGHT over SEARCH_XY,
        looking straight down
    2   find the plant: the one nearest SEARCH_XY, its size and height (no
        motion)
    3   to the observation pose: OBSERVE_HEIGHT over the plant, looking
        straight down, the plant clear of the cutter in the picture
    4   look: the leaves, a frame each, the leaf with the most room at its
        petiole (no motion)
    5   closer: the nozzle turned to the chosen leaf's J -> C, kept straight
        down (the pitch is step 6's), the stereo pair RELOOK_HEIGHT over the
        plant's top, the leaf shifted clear of the cutter in the picture; at
        NEAR_SPEED
    6   look again from there: the same leaf, its frame refreshed -- stops
        when it moved past RELOOK_MAX_MOVE (no motion)
    7   onto the leaf: straight to RELOOK_OFFSET over its C, straight down
        to APPROACH_OFFSET
    -- the take, only when DESCEND_DIST > 0 (a dry run, 0, skips it: the
       nozzle stays APPROACH_OFFSET over the leaf, no vacuum, no blade) --
    8   vacuum on, down along the leaf's z SEAL_STEP at a time until the
        pressure sensor reads the nozzle sealed (at most DESCEND_DIST),
        SUCK_WAIT
    9   ultrasonic on, US_SPINUP
    10  deploy the blade: the cut; CUT_DWELL
    11  ultrasonic off
    
    12  retract the blade
    -- then --
    13  lift LIFT straight up, the leaf held: LIFT_CLEAR at the leaf's
        tilt, then the nozzle turned straight down on the way
    14  over the capper's tube: STORE_CLEARANCE over its top, nozzle down
    15  down into the tube: STORE_CLEARANCE + STORE_DEPTH
    16  blow BLOW_TIME, vacuum off: the leaf drops in
    17  back up out of the tube

Each step waits for 'g' (q / Esc / Ctrl+C stop); ``--c`` runs straight
through (q stops before the next step). A motion is planned before its 'g'
and printed. Whatever ends the run, the ultrasonic is switched off and the
blade's board closed -- which also stops the vacuum (a held leaf drops).
The vision steps are ``coripps.vision.sampler_stereo.leaf_pipeline``'s
(frames saved to its ``observe/``); the collision model is the sampler,
the picker (read once, frozen) and the ground -- not the plant, the
capper or the tube.

    py -3.12 coripps/tool/leaf/suc_cut_store.py
    py -3.12 coripps/tool/leaf/suc_cut_store.py --c
"""

import sys
import time
import argparse

import numpy as np

import one.utils.math as oum
from one.drivers.cobotta import BCapError, CobottaMotion

from coripps.end_effectors.xh430_cutter.exec import XH430CutterExec
from coripps.tool.home_sampler import _build, _SAMPLER_HOST
from coripps.tool.keys import _quit_pressed, _wait_for_g
from coripps.vision.sampler_stereo import leaf_pipeline as lp
from coripps.vision.sampler_stereo import leaves as lv


# ---- the cut (m / s) -------------------------------------------------------
DESCEND_DIST = 0.0              # at most this past APPROACH_OFFSET, down
                                # until sealed; 0 = dry run, no touch
SEAL_STEP = 0.0005              # down a step at a time ...
SEAL_WAIT = 0.15                # s, ... the sensor read after each
SEAL_SPEED = 5.0                # %
SUCK_WAIT = 1.0                 # s, held on once sealed
US_SPINUP = 0.5                 # ultrasonic on before the blade moves
CUT_DWELL = 0.5                 # blade out, ultrasonic still on
LIFT = 0.100                    # straight up after the cut ...
LIFT_CLEAR = 0.010              # ... its first part at the leaf's tilt
# ---- the store: the tube held in the capper (cell world frame) --------------
TUBE_XY = (0.2535, -0.04)       # the tube's axis
TUBE_TOP_Z = 0.090              # its open top -- estimated, MEASURE IT
STORE_CLEARANCE = 0.040         # nozzle over the top before going down
STORE_DEPTH = 0.0               # nozzle below the top -- 0: at the opening
BLOW_TIME = 1.0
ROLL_STEP_DEG = 15              # the nozzle's free roll over the tube
# ---- speeds (%) --------------------------------------------------------------
FREE_SPEED = 30.0               # search, observe, to the plant / capper --
                                # low for the first runs
NEAR_SPEED = 10.0               # straight moves at the plant and the tube
STRAIGHT_STEP = lp.DESCENT_STEP


class LeafSession:
    """The sampler, its cutter (blade, ultrasonic, vacuum), the stereo pair
    and the vision models; the planning state between steps."""

    def __init__(self):
        self.tf_flange_cam = lp.load_handeye()
        print('loading SAM 3 ...')
        self.segmenter = lv.Segmenter()
        _, _, self.sampler, self.pln_ctx = _build()     # freezes the picker
        self.cobotta = CobottaMotion(_SAMPLER_HOST, speed=FREE_SPEED).connect()
        try:
            self.cutter = XH430CutterExec().connect()
        except Exception:
            self.cobotta.disconnect()
            raise
        try:
            self.eye = lp.Eye()
        except Exception:
            self.cutter.disconnect()
            self.cobotta.disconnect()
            raise
        self.leaf = None
        self.plant = None

    def close(self):
        try:
            self.cutter.ultrasonic(False)
        finally:
            self.eye.close()
            self.cutter.disconnect()    # all DIO off: blade, vacuum
            self.cobotta.disconnect()   # motors off, arm given back

    # -- helpers ----------------------------------------------------------
    def qs_now(self):
        return self.pln_ctx.enforce_bounds(
            self.cobotta.joint_angles(ndof=self.sampler.ndof))

    def run(self, path, speed, what):
        """Play a planned joint path from where the arm is."""
        off = np.max(np.abs(self.qs_now() - path[0]))
        if off > lp.ARRIVAL_TOLERANCE:
            raise RuntimeError(f'{what}: the arm moved since planning '
                               f'({np.degrees(off):.2f} deg)')
        self.cobotta.enable(speed=speed)
        self.cobotta.move_path(path[1:] if len(path) > 1 else path)

    def line(self, delta, what, upright=False, qs=None, step=STRAIGHT_STEP):
        """A straight tcp path by ``delta`` (world, m) from where the arm
        is (or from joints ``qs``), a waypoint each ``step``, the tcp's
        orientation kept -- or, ``upright``, turned on the way to the
        nozzle straight down (the least turn)."""
        qs = self.qs_now() if qs is None else qs
        self.sampler.fk(qs=qs)
        cur = np.array(self.sampler.cutter.tcp('tcp').tf, dtype=np.float64)
        rot = cur[:3, :3]
        n = max(int(np.ceil(np.linalg.norm(delta) / step)), 1)
        if upright:
            axis = np.cross(rot[:, 2], [0.0, 0.0, -1.0])
            turn = np.arctan2(np.linalg.norm(axis), -rot[2, 2])
            if turn > 1e-6:
                end = oum.rotmat_from_axangle(axis / np.linalg.norm(axis),
                                              turn) @ rot
                n = max(n, int(np.ceil(turn / lp.TURN_STEP)))
                rot = np.array(oum.rotmat_slerp(rot, end, n + 1)[1:])
        path = lp.straight(self.sampler, self.pln_ctx, qs,
                           [cur[:3, 3] + delta * k / n for k in range(1, n + 1)],
                           rot, what)
        if path is None:
            raise RuntimeError(f'{what}: not plannable')
        return path

    # -- the steps: each returns (description, run) -- run after the 'g' --
    def s_search(self):
        start = self.qs_now()
        goal = lp.search_goal(self.sampler, self.pln_ctx, self.tf_flange_cam,
                              self.eye, start)
        path = None if goal is None else lp.plan(self.pln_ctx, start, goal)
        if path is None:
            raise RuntimeError('search: no collision-free pose / path')
        return (f'to the search pose: {lp.SEARCH_HEIGHT * 1000:.0f} mm over '
                f'{np.round(np.array(lp.SEARCH_XY) * 1000)} mm '
                f'({len(path)} waypoints)',
                lambda: self.run(path, FREE_SPEED, 'search'))

    def s_find_plant(self):
        def run():
            time.sleep(lp.SETTLE)
            self.plant = lp.find_plant(self.eye, self.sampler, self.qs_now(),
                                       self.tf_flange_cam, self.segmenter)
            if self.plant is None:
                raise RuntimeError('find the plant: no plant found')
            print(f"plant: centre "
                  f"{np.round(self.plant['center'][:2] * 1000, 1)} mm, "
                  f"radius {self.plant['radius'] * 1000:.1f} mm, top z "
                  f"{self.plant['plant_z'] * 1000:.1f} mm")
        return 'find the plant nearest the search point', run

    def s_observe(self):
        start = self.qs_now()
        goal = lp.observe_goal(self.sampler, self.pln_ctx, self.tf_flange_cam,
                               self.eye, self.plant, start)
        path = None if goal is None else lp.plan(self.pln_ctx, start, goal)
        if path is None:
            raise RuntimeError('observe: no collision-free pose / path')
        return (f'to the observation pose: {lp.OBSERVE_HEIGHT * 1000:.0f} mm '
                f'over the plant ({len(path)} waypoints)',
                lambda: self.run(path, FREE_SPEED, 'observe'))

    def s_look(self):
        def run():
            time.sleep(lp.SETTLE)
            found = lp.look(self.eye, self.sampler, self.qs_now(),
                            self.tf_flange_cam, self.segmenter, self.plant)
            if found is None:
                raise RuntimeError('look: no leaves found')
            leaves, chosen = found
            self.leaf = leaves[chosen]
            print(f"look: {len(leaves)} leaves, chosen leaf {chosen} -- C "
                  f"{np.round(self.leaf['C'] * 1000, 1)} mm, room at J "
                  f"{self.leaf['room'][0]:.1f} / {self.leaf['room'][1]:.1f} "
                  'mm')
        return 'look: the leaves and the one to take', run

    def s_closer(self):
        start = self.qs_now()
        goal = lp.relook_goal(self.sampler, self.pln_ctx, self.tf_flange_cam,
                              self.eye, self.leaf, self.plant['plant_z'], start)
        path = None if goal is None else lp.plan(self.pln_ctx, start, goal)
        if path is None:
            raise RuntimeError('closer: no collision-free pose / path')
        return (f'closer: the nozzle turned to the leaf, level, the eye '
                f'{lp.RELOOK_HEIGHT * 1000:.0f} mm over the plant, the leaf '
                f'clear of the cutter ({len(path)} waypoints, '
                f'{NEAR_SPEED:.0f}%)',
                lambda: self.run(path, NEAR_SPEED, 'closer'))

    def s_relook(self):
        def run():
            time.sleep(lp.SETTLE)
            new = lp.relook(self.eye, self.sampler, self.qs_now(),
                            self.tf_flange_cam, self.segmenter, self.leaf)
            if new is None:
                raise RuntimeError('look again: the leaf is not found')
            self.leaf = new
        return 'look again from there: the leaf refreshed', run

    def s_onto_leaf(self):
        path = lp.to_leaf(self.sampler, self.pln_ctx, self.leaf['tf'],
                          self.qs_now())
        if path is None:
            raise RuntimeError('onto the leaf: not plannable')
        return (f'onto the leaf: over its C, then straight down to '
                f'{lp.APPROACH_OFFSET * 1000:.0f} mm over it',
                lambda: self.run(path, NEAR_SPEED, 'onto the leaf'))

    def s_seal(self):
        """Vacuum on, then down along the leaf's z a SEAL_STEP at a time
        until the pressure sensor reads the nozzle sealed -- the leaf's
        height is a stereo guess (mm off), the sensor finds it -- at most
        DESCEND_DIST; not sealed by then: stopped."""
        path = self.line(DESCEND_DIST * self.leaf['tf'][:3, 2], 'seal',
                         step=SEAL_STEP)

        def run():
            off = np.max(np.abs(self.qs_now() - path[0]))
            if off > lp.ARRIVAL_TOLERANCE:
                raise RuntimeError(f'seal: the arm moved since planning '
                                   f'({np.degrees(off):.2f} deg)')
            self.cutter.vacuum('suck')
            time.sleep(SEAL_WAIT)
            self.cobotta.enable(speed=SEAL_SPEED)
            for k, qs in enumerate(path):
                if k:
                    self.cobotta.move_path([qs])
                    time.sleep(SEAL_WAIT)
                if self.cutter.sealed():
                    print(f'seal: sealed {k * SEAL_STEP * 1000:.1f} mm past '
                          f'{lp.APPROACH_OFFSET * 1000:.0f} mm over the leaf')
                    time.sleep(SUCK_WAIT)
                    return
            raise RuntimeError(f'seal: not sealed {DESCEND_DIST * 1000:.0f} '
                               f'mm past {lp.APPROACH_OFFSET * 1000:.0f} mm '
                               'over the leaf')
        return (f'vacuum on, down until sealed (at most '
                f'{DESCEND_DIST * 1000:.0f} mm, {SEAL_STEP * 1000:.1f} mm a '
                'step)', run)

    def s_us_on(self):
        def run():
            self.cutter.ultrasonic(True)
            time.sleep(US_SPINUP)
        return f'ultrasonic on, {US_SPINUP:.1f} s', run

    def s_cut(self):
        def run():
            self.cutter.deploy()
            time.sleep(CUT_DWELL)
        return f'deploy the blade: the cut, {CUT_DWELL:.1f} s', run

    def s_us_off(self):
        return 'ultrasonic off', lambda: self.cutter.ultrasonic(False)

    def s_retract(self):
        return 'retract the blade', lambda: self.cutter.retract()

    def s_lift(self):
        # first LIFT_CLEAR at the leaf's tilt, off the plant; then turned
        # upright on the way -- kept at the tilt, the wrist bends the
        # cutter's base into the forearm ~85 mm up
        # -- unless the wrist cannot turn that way (its last joint at its
        # end): then all of it at the tilt, if that is collision-free
        clear = self.line(np.array([0.0, 0.0, LIFT_CLEAR]), 'lift off')
        try:
            rest = self.line(np.array([0.0, 0.0, LIFT - LIFT_CLEAR]), 'lift',
                             upright=True, qs=clear[-1])
            how = 'then the nozzle turned straight down on the way'
        except RuntimeError:
            rest = self.line(np.array([0.0, 0.0, LIFT - LIFT_CLEAR]), 'lift',
                             qs=clear[-1])
            how = 'and the rest too (the wrist cannot turn it upright)'
        path = clear + rest[1:]
        return (f'lift {LIFT * 1000:.0f} mm straight up, the leaf held: '
                f'{LIFT_CLEAR * 1000:.0f} mm at its tilt, {how}',
                lambda: self.run(path, NEAR_SPEED, 'lift'))

    def s_to_capper(self):
        start = self.qs_now()
        pos = np.array([*TUBE_XY, TUBE_TOP_Z + STORE_CLEARANCE])
        tcp = self.sampler.cutter.tcp('tcp')
        best = None
        for roll in range(0, 360, ROLL_STEP_DEG):
            a = np.deg2rad(roll)
            x = np.array([np.cos(a), np.sin(a), 0.0])
            z = np.array([0.0, 0.0, -1.0])       # the nozzle down
            rot = np.column_stack([x, np.cross(z, x), z])
            qs = lp.nearest_valid(self.pln_ctx, self.sampler.ik(
                pos.astype(np.float32), rot.astype(np.float32), tcp=tcp,
                ref_qs=start), start)
            if qs is not None and (best is None or np.max(np.abs(qs - start))
                                   < np.max(np.abs(best - start))):
                best = qs
        path = None if best is None else lp.plan(self.pln_ctx, start, best)
        if path is None:
            raise RuntimeError('to the capper: no collision-free pose / path')
        return (f'over the capper\'s tube: {np.round(pos * 1000, 1)} mm, '
                f'nozzle down ({len(path)} waypoints)',
                lambda: self.run(path, FREE_SPEED, 'to the capper'))

    def s_into_tube(self):
        dz = STORE_CLEARANCE + STORE_DEPTH
        path = self.line(np.array([0.0, 0.0, -dz]), 'into the tube')
        return (f'down {dz * 1000:.0f} mm into the tube',
                lambda: self.run(path, NEAR_SPEED, 'into the tube'))

    def s_blow(self):
        def run():
            self.cutter.vacuum('blow')
            time.sleep(BLOW_TIME)
            self.cutter.vacuum('off')
        return f'blow {BLOW_TIME:.1f} s, vacuum off: the leaf drops in', run

    def s_out_of_tube(self):
        dz = STORE_CLEARANCE + STORE_DEPTH
        path = self.line(np.array([0.0, 0.0, dz]), 'out of the tube')
        return (f'back up {dz * 1000:.0f} mm out of the tube',
                lambda: self.run(path, NEAR_SPEED, 'out of the tube'))

    def steps(self):
        """The step makers, in order (each planned when its turn comes)."""
        steps = [self.s_search, self.s_find_plant, self.s_observe, self.s_look,
                 self.s_closer, self.s_relook, self.s_onto_leaf]
        if DESCEND_DIST > 0:
            # the take: only when it touches -- a dry run's blade would
            # swing APPROACH_OFFSET over the leaves, its vacuum pull them
            steps += [self.s_seal, self.s_us_on, self.s_cut, self.s_us_off,
                      self.s_retract]
        return steps + [self.s_lift, self.s_to_capper, self.s_into_tube,
                        self.s_blow, self.s_out_of_tube]


def main(continuous=False):
    try:
        session = LeafSession()
    except Exception as exc:
        print(f'cannot start: {exc}')
        return False
    try:
        makers = session.steps()
        for k, make in enumerate(makers, 1):
            what, run = make()
            title = f'step {k}/{len(makers)}: {what}'
            if continuous:
                print(f'\n{title}')
                if _quit_pressed():
                    print('stopped')
                    return False
            elif not _wait_for_g(title):
                print('stopped')
                return False
            run()
        print('\ndone: the leaf is in the tube')
        return True
    except (RuntimeError, OSError, BCapError) as exc:
        print(f'stopped: {exc}')
        return False
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--c', action='store_true',
                        help='continuous: no g per step (q stops)')
    args = parser.parse_args()
    sys.exit(0 if main(continuous=args.c) else 1)
