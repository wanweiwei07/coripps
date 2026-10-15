"""Suck, cut and store one leaf, step by step: observe the plant with the
cutter hand's D405, line the cutter up with the chosen leaf, take it with the
nozzle, cut its petiole with the ultrasonic blade, lift it, and put it into
the tube in the capper.

    1   to the observation pose (the D405 OBSERVE_HEIGHT over VIEW_CENTER)
    2   look: the leaves, a frame each, the leaf with the most room at its
        petiole (no motion)
    3   over the plant centre: PRE_APPROACH_OFFSET over it, straight down
        to RELOOK_OFFSET
    4   look again from there: the same leaf, its frame refreshed (no motion)
    5   onto the leaf: straight to RELOOK_OFFSET over its C, straight down
        to APPROACH_OFFSET
    6   down DESCEND_DIST more along the leaf's z -- 0 for a dry run: the
        nozzle stays APPROACH_OFFSET over the leaf (skipped when 0)
    7   vacuum on (suck), SUCK_WAIT
    8   ultrasonic on, US_SPINUP
    9   deploy the blade: the cut; CUT_DWELL
    10  ultrasonic off
    11  retract the blade
    12  lift LIFT straight up, the leaf held
    13  over the capper's tube: STORE_CLEARANCE over its top, nozzle down
    14  down into the tube: STORE_CLEARANCE + STORE_DEPTH
    15  blow BLOW_TIME, vacuum off: the leaf drops in
    16  back up out of the tube

Each step waits for 'g' (q / Esc / Ctrl+C stop); ``--c`` runs straight
through (q stops before the next step). A motion is planned before its 'g'
and printed. Whatever ends the run, the ultrasonic is switched off and the
blade's board closed -- which also stops the vacuum (a held leaf drops).
The vision steps are ``coripps.vision.d405.leaf_pipeline``'s; the collision
model is the sampler, the picker (read once, frozen) and the ground -- not
the plant, the capper or the tube.

    py -3.12 coripps/tool/leaf/suc_cut_store.py
    py -3.12 coripps/tool/leaf/suc_cut_store.py --c
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np

from one.drivers.cobotta import BCapError, CobottaMotion

from coripps.end_effectors.xh430_cutter.exec import XH430CutterExec
from coripps.tool.home_sampler import _build, _SAMPLER_HOST
from coripps.tool.keys import _quit_pressed, _wait_for_g
from coripps.vision.d405 import leaf_pipeline as lp
from coripps.vision.d405 import leaves as lv
from coripps.vision.d405.camera import D405


# ---- the cut (m / s) -------------------------------------------------------
DESCEND_DIST = 0.0              # past APPROACH_OFFSET; 0 = dry run, no touch
SUCK_WAIT = 1.0
US_SPINUP = 0.5                 # ultrasonic on before the blade moves
CUT_DWELL = 0.5                 # blade out, ultrasonic still on
LIFT = 0.100                    # straight up after the cut
# ---- the store: the tube held in the capper (cell world frame) --------------
TUBE_XY = (0.2535, -0.04)       # the tube's axis
TUBE_TOP_Z = 0.090              # its open top -- estimated, MEASURE IT
STORE_CLEARANCE = 0.040         # nozzle over the top before going down
STORE_DEPTH = 0.0               # nozzle below the top -- 0: at the opening
BLOW_TIME = 1.0
ROLL_STEP_DEG = 15              # the nozzle's free roll over the tube
# ---- speeds (%) --------------------------------------------------------------
FREE_SPEED = 100.0              # observe, over the plant, to the capper
NEAR_SPEED = 10.0               # straight moves at the plant and the tube
STRAIGHT_STEP = lp.DESCENT_STEP


class LeafSession:
    """The sampler, its cutter (blade, ultrasonic, vacuum), the D405 and the
    vision models; the planning state between steps."""

    def __init__(self):
        self.tf_flange_cam = lp.load_handeye()
        print('loading SAM 3 ...')
        self.segmenter = lv.Segmenter()
        _, _, self.sampler, self.pln_ctx = _build()     # freezes the picker
        self.cobotta = CobottaMotion(_SAMPLER_HOST, speed=FREE_SPEED).connect()
        try:
            self.cutter = XH430CutterExec().connect()
        except Exception:
            self.cobotta.close()
            raise
        self.leaf = None
        self.plant_tf = None

    def close(self):
        try:
            self.cutter.ultrasonic(False)
        finally:
            self.cutter.close()         # all DIO off: blade, vacuum
            self.cobotta.close()        # motors off, arm given back

    # -- helpers ----------------------------------------------------------
    def qs_now(self):
        return self.pln_ctx.enforce_bounds(
            self.cobotta.joint_angles(ndof=self.sampler.ndof))

    def tcp_now(self):
        self.sampler.fk(qs=self.qs_now())
        return np.array(self.sampler.cutter.tcp('tcp').tf, dtype=np.float64)

    def run(self, path, speed, what):
        """Play a planned joint path from where the arm is."""
        off = np.max(np.abs(self.qs_now() - path[0]))
        if off > lp.ARRIVAL_TOLERANCE:
            raise RuntimeError(f'{what}: the arm moved since planning '
                               f'({np.degrees(off):.2f} deg)')
        self.cobotta.enable(speed=speed)
        self.cobotta.move_path(path[1:] if len(path) > 1 else path)

    def line(self, delta, what):
        """A straight tcp path by ``delta`` (world, m) from where the arm
        is, the tcp's orientation kept."""
        qs = self.qs_now()
        cur = self.tcp_now()
        n = max(int(np.ceil(np.linalg.norm(delta) / STRAIGHT_STEP)), 1)
        path = lp.straight(self.sampler, self.pln_ctx, qs,
                           [cur[:3, 3] + delta * k / n for k in range(1, n + 1)],
                           cur[:3, :3], what)
        if path is None:
            raise RuntimeError(f'{what}: not plannable')
        return path

    def frame(self):
        time.sleep(lp.SETTLE)
        cam = D405()
        try:
            color, depth, _ = cam.frames()
            return color, depth, cam.K
        finally:
            cam.close()

    # -- the steps: each returns (description, run) -- run after the 'g' --
    def s_observe(self):
        start = self.qs_now()
        goal = lp.observe_goal(self.sampler, self.pln_ctx, self.tf_flange_cam,
                               start)
        path = None if goal is None else lp.plan(self.pln_ctx, start, goal)
        if path is None:
            raise RuntimeError('observe: no collision-free pose / path')
        return (f'to the observation pose ({len(path)} waypoints)',
                lambda: self.run(path, FREE_SPEED, 'observe'))

    def s_look(self):
        def run():
            color, depth, K = self.frame()
            qs = self.qs_now()
            lp.save_frame(color, depth, K, qs)  # vision/d405/observe/
            cam = lp.tf_w_cam(self.sampler, qs, self.tf_flange_cam)
            found = lv.leaf_frames(color, depth, K, cam, self.segmenter)
            if found is None or not found[0]:
                raise RuntimeError('look: no leaves found')
            leaves, rosette = found
            chosen = lv.choose(leaves)
            os.makedirs(lp.OBSERVE_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(lp.OBSERVE_DIR, 'leaves.png'),
                        lv.draw(color, leaves, rosette, chosen))
            self.leaf = leaves[chosen]
            self.plant_tf = lp.plant_frame(self.leaf, rosette, K, cam)
            print(f"look: {len(leaves)} leaves, chosen leaf {chosen} -- C "
                  f"{np.round(self.leaf['C'] * 1000, 1)} mm, room at J "
                  f"{self.leaf['room'][0]:.1f} / {self.leaf['room'][1]:.1f} "
                  f"mm; plant centre {np.round(self.plant_tf[:3, 3] * 1000, 1)}"
                  ' mm')
        return 'look: the leaves and the one to take', run

    def s_over_plant(self):
        path = lp.to_plant(self.sampler, self.pln_ctx, self.plant_tf,
                           self.qs_now())
        if path is None:
            raise RuntimeError('over the plant: not plannable')
        return (f'over the plant centre: {lp.PRE_APPROACH_OFFSET * 1000:.0f} '
                f'mm, then straight down to {lp.RELOOK_OFFSET * 1000:.0f} mm',
                lambda: self.run(path, FREE_SPEED, 'over the plant'))

    def s_relook(self):
        def run():
            job = {'sampler': self.sampler, 'segmenter': self.segmenter,
                   'tf_flange_cam': self.tf_flange_cam}
            new, _ = lp.relook(job, self.leaf)
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

    def s_descend(self):
        delta = DESCEND_DIST * self.leaf['tf'][:3, 2]
        path = self.line(delta, 'descend')
        return (f'down {DESCEND_DIST * 1000:.1f} mm more along the leaf z',
                lambda: self.run(path, NEAR_SPEED, 'descend'))

    def s_suck(self):
        def run():
            self.cutter.vacuum('suck')
            time.sleep(SUCK_WAIT)
        return f'vacuum on (suck), {SUCK_WAIT:.1f} s', run

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
        path = self.line(np.array([0.0, 0.0, LIFT]), 'lift')
        return (f'lift {LIFT * 1000:.0f} mm straight up, the leaf held',
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
        steps = [self.s_observe, self.s_look, self.s_over_plant,
                 self.s_relook, self.s_onto_leaf]
        if DESCEND_DIST > 0:
            steps.append(self.s_descend)
        return steps + [self.s_suck, self.s_us_on, self.s_cut, self.s_us_off,
                        self.s_retract, self.s_lift, self.s_to_capper,
                        self.s_into_tube, self.s_blow, self.s_out_of_tube]


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
