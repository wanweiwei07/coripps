"""Suck, cut and store one leaf, step by step: find the plant and look
at it with the sampler's stereo cameras, line the cutter up with the
chosen leaf, take it with the nozzle, cut its petiole with the ultrasonic
blade, lift it, and put it into the tube in the capper.

    1   to the search pose: the stereo pair SEARCH_HEIGHT over SEARCH_XY,
        looking straight down
    2   find the plant: the one nearest SEARCH_XY, its size and height (no
        motion)
    3   to the observation pose, near: the nozzle's tip LOOK_CLEARANCE
        over the plant's top
        (the whole plant and its pot in the picture), looking straight
        down, the plant clear of the cutter in the picture
    4   look, once: the leaves, a frame each, the leaf with the most room
        at its petiole -- its frame the one the take goes by (no motion)
    5   to the close look: the nozzle turned to the leaf's J -> C, the
        eye CLOSE_LOOK_HEIGHT over the plant's top, shifted so the leaf
        is beside the cutter in the picture
    6   close look: the chosen leaf again, its frame (C, J, height) the
        one the take goes by (no motion)
    7   onto the leaf: straight to RELOOK_OFFSET over its C, straight down
        to APPROACH_OFFSET
    -- the take, only when DESCEND_DIST > 0 (a dry run, 0, skips it: the
       nozzle stays APPROACH_OFFSET over the leaf, no vacuum, no blade) --
    8   vacuum on, down along the leaf's z SEAL_STEP at a time until the
        pressure sensor reads the nozzle sealed (at most DESCEND_DIST),
        SUCK_WAIT
    9   tilt: the held leaf up to TILT_TO_DEG about the plant's centre
        (C and J up) -- off the soil and the pot's rim
    10  ultrasonic on, US_SPINUP
    11  the cut: XH430CutterExec.cut -- the blade's last stretch slow, a
        step at a time; the leaf checked after each pass
    12  ultrasonic off
    13  retract the blade
    -- then --
    14  lift LIFT straight up, the leaf held: LIFT_CLEAR at the leaf's
        tilt, then the nozzle turned straight down on the way
    15  over the capper's tube: STORE_CLEARANCE over its top, nozzle down
    16  down into the tube: STORE_CLEARANCE + STORE_DEPTH
    17  blow BLOW_TIME, vacuum off: the leaf drops in
    18  back up out of the tube
    -- the take failed (the leaf not seen again, not sealed, not cut,
       the leaf lost): back up, look again, another leaf; MAX_TRIES in
       all, then given up

Each step waits for 'g' (q / Esc / Ctrl+C stop; in the console, the
viewer); ``--c`` runs straight through (q stops
before the next step). The live view (``coripps.monitor``) runs beside:
the arms where they are, the last look's point cloud, and a panel in it
-- the left eye's picture: the vision steps' leaves (the chosen green, the candidates
yellow, the others grey) with a table of every leaf's numbers and why it
is not taken; after a motion the left eye now with the plant's circle and
the chosen leaf's C -> J where they are. Closing the viewer stops before
the next step. A motion is planned before its 'g'
and printed. Whatever ends the run, the ultrasonic is switched off and the
blade's board closed -- which also stops the vacuum (a held leaf drops).
The vision steps are ``coripps.vision.sampler_stereo.leaf_pipeline``'s
(frames saved to its ``observe/``); the collision model is the sampler,
the picker (read once, frozen) and the ground -- not the plant, the
capper or the tube.

    py -3.12 coripps/tool/leaf/suc_cut_store.py
    py -3.12 coripps/tool/leaf/suc_cut_store.py --c
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np

import one.utils.math as oum
from one.drivers.cobotta import BCapError, CobottaMotion

import coripps.end_effectors.xh430_cutter.exec as xce
from coripps.end_effectors.xh430_cutter.exec import XH430CutterExec
from coripps.tool.home_sampler import _build, _SAMPLER_HOST
from coripps.robots.mobisys import capper_tube_pos
from coripps import monitor
from coripps.tool.keys import _QUIT_KEYS, _quit_pressed, _wait_for_g
from coripps.vision.sampler_stereo import leaf_pipeline as lp
from coripps.vision.sampler_stereo import leaves as lv


# ---- the cut (m / s) -------------------------------------------------------
DESCEND_DIST = 0.008          # at most this past APPROACH_OFFSET, down
                                # until sealed; 0 = dry run, no touch
SEAL_STEP = 0.0005              # down a step at a time ...
SEAL_WAIT = 0.15                # s, ... the sensor read after each
SEAL_SPEED = 5.0                # %
SUCK_WAIT = 1.0                 # s, held on once sealed
# the descent centred on the leaf by the left eye (s_seal)
SERVO_UNTIL = 0.004             # m over the leaf's height: below, blind
SERVO_GAIN = 0.5                # of what is off (the last SERVO_AVG
SERVO_AVG = 3                   # measures' mean), moved each step ...
SERVO_MAX = 0.006               # ... at most this in all (m)
SERVO_DEADBAND = 0.0005         # m: less is left alone
SERVO_SEARCH = 0.004            # m: the leaf looked for this far off
SERVO_MIN_SEEN = 0.3            # of its outline seen past the nozzle, ...
SERVO_MIN_MATCH = 0.5           # ... matched at least this well; else blind
SERVO_GREEN = 15                # green: G over R and over B by this
VAC_ON_ABOVE = 0.004            # m over the leaf's height: the vacuum on
                                # from here down (not from APPROACH_OFFSET)
# the nozzle's pitch on the way in: the leaf's slope up J -> C (square to
# it), at most PITCH_MAX_DEG; drooping: 0 (level)
PITCH_MAX_DEG = 35.0
TILT_TO_DEG = 15.0              # held, the leaf tilted up about the
                                # plant's centre to this, when less -- before
                                # the cut
TILT_SPEED = 5.0                # %
US_SPINUP = 0.5                 # ultrasonic on before the blade moves
                                # the cut itself: XH430CutterExec.cut (its
                                # CUT_* in the cutter's exec.py)
LIFT = 0.100                    # straight up after the cut ...
LIFT_CLEAR = 0.010              # ... its first part at the leaf's tilt
# ---- the store: the tube held in the capper (cell world frame) --------------
# the tube's axis and top: the capper's 'tube' tcp (mobisys.capper_tube_pos,
# its cap's top as placed); the open top, the cap flipped off, taken there
# too -- CHECK IT on the cell
# the nozzle seen over the hole's edge at the model's axis (2026-11-19):
# moved by STORE_XY_FIX (m, world) -- the sampler's own offset, the picker's
# use of the capper's 'tube' tcp left alone -- (0.001, 0.008) then; zeroed
# once the nozzle's tcp was calibrated (xh430_cutter._TCP, 2026-11-19)
STORE_XY_FIX = (0.0, 0.0)
TUBE_XY = tuple(capper_tube_pos()[:2] + np.array(STORE_XY_FIX))
TUBE_TOP_Z = float(capper_tube_pos()[2])
STORE_CLEARANCE = 0.040         # nozzle over the top before going down
STORE_DEPTH = 0.0               # nozzle below the top -- 0: at the opening
BLOW_TIME = 0.1                 # s: a puff -- longer blows the leaf back out
ROLL_STEP_DEG = 15              # the nozzle's free roll over the tube
# ---- speeds (%) --------------------------------------------------------------
FREE_SPEED = 30.0               # search, observe, to the plant / capper --
                                # low for the first runs
NEAR_SPEED = 10.0               # straight moves at the plant and the tube
STRAIGHT_STEP = lp.DESCENT_STEP
# the arm found this far (deg) off where a path was planned from: the path
# run from where it is, when that first stretch is free; further: stopped
DRIFT_MAX_DEG = 3.0
# the one look, near: the nozzle's tip this far over the plant's top (the
# eye ~50 mm further up, behind it: ~80 mm, past the stereo's nearest
# ~61 mm -- f 299 px x baseline 26 mm / LEAF_MAX_DISP 128 px) -- it sees the
# whole plant and its pot from there; no look from afar to agree with
LOOK_CLEARANCE = 0.030
# the chosen leaf looked at again, closer, before the take: the eye this far
# over the plant's top, beside the cutter (at the approach the nozzle hides
# the leaf, and 60 mm is past the stereo's disparity range)
CLOSE_LOOK_HEIGHT = 0.080


def leaf_plane(leaf):
    """``(a, b, c)``: the leaf's plane z = a x + b y + c (world), fit to its
    depth points (the worst residuals dropped, once); None when too few."""
    pts = np.asarray(leaf['blade_pts'], np.float64)
    if len(pts) < 10:
        return None
    keep = np.ones(len(pts), bool)
    for _ in range(2):
        a = np.c_[pts[keep, :2], np.ones(keep.sum())]
        coef = np.linalg.lstsq(a, pts[keep, 2], rcond=None)[0]
        res = np.abs(np.c_[pts[:, :2], np.ones(len(pts))] @ coef - pts[:, 2])
        keep = res <= max(2.5 * np.median(res), 1e-4)
    return coef


def on_plane(origin, ray, plane):
    """Where the line ``origin`` + t ``ray`` meets ``plane`` (a, b, c)."""
    a, b, c = plane
    t = (a * origin[0] + b * origin[1] + c - origin[2]) \
        / (ray[2] - a * ray[0] - b * ray[1])
    return origin + t * ray


def leaf_pitch(leaf):
    """The leaf's slope along its J -> C (rad, + the C side up), from
    :func:`leaf_plane`; None when too few points."""
    coef = leaf_plane(leaf)
    if coef is None:
        return None
    y = np.array([leaf['tf'][0, 1], leaf['tf'][1, 1]])
    y /= np.linalg.norm(y)
    return float(np.arctan(coef[:2] @ y))


def pitched(tf, pitch):
    """The leaf frame ``tf`` with its y (J -> C) pitched ``pitch`` (rad,
    + up) about its x, the nozzle (z) square to it -- 0: level, z straight
    down."""
    tf = np.array(tf, dtype=np.float64)
    y = np.array([tf[0, 1], tf[1, 1], 0.0])
    y /= np.linalg.norm(y)
    x = np.cross([0.0, 0.0, 1.0], y)
    y = np.cos(pitch) * y + np.sin(pitch) * np.array([0.0, 0.0, 1.0])
    tf[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
    return tf


MAX_TRIES = 3                   # takes of a plant; all failed: given up


class TakeFailed(RuntimeError):
    """The take did not work -- not sealed, the blade stalled (not cut), or
    the leaf lost: back up, look again, take again."""


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
        self.lifted = True              # not down at a leaf
        self.leaves = []                # the last look's leaves
        self.target = None              # (x, y) m: the leaf to take, asked
                                        # for (nearest C; rules not applied)
        self.tried = []                 # the Cs of the leaves failed: the
                                        # next look takes another

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
        now = self.qs_now()
        offs = np.abs(now - path[0])
        off = np.max(offs)
        if off > lp.ARRIVAL_TOLERANCE:
            # a little (the arm settling, a joint pulled by its cable while
            # the motors were off): from where it is, when that is free
            j = int(np.argmax(offs)) + 1
            nxt = path[1] if len(path) > 1 else path[0]
            if (off <= np.deg2rad(DRIFT_MAX_DEG)
                    and self.pln_ctx.is_motion_valid(now, nxt)):
                print(f'{what}: the arm {np.degrees(off):.2f} deg off where '
                      f'it was planned from (J{j}) -- going on from there')
                path = [now] + list(path[1:])
            else:
                raise RuntimeError(f'{what}: the arm moved since planning '
                                   f'({np.degrees(off):.2f} deg, J{j})')
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
            qs = self.qs_now()
            self.plant = lp.find_plant(self.eye, self.sampler, qs,
                                       self.tf_flange_cam, self.segmenter)
            self.publish_look(qs, 'find the plant: ' + (
                'none' if self.plant is None else
                f"radius {self.plant['radius'] * 1000:.0f} mm, top z "
                f"{self.plant['plant_z'] * 1000:.0f} mm"))
            if self.plant is None:
                raise RuntimeError('find the plant: no plant found')
            print(f"plant: centre "
                  f"{np.round(self.plant['center'][:2] * 1000, 1)} mm, "
                  f"radius {self.plant['radius'] * 1000:.1f} mm, top z "
                  f"{self.plant['plant_z'] * 1000:.1f} mm")
        return 'find the plant nearest the search point', run

    def eye_behind_tip(self, qs):
        """How far the eye sits behind the nozzle's tip along its axis
        (m; the two look the same way)."""
        self.sampler.fk(qs=qs)
        tf = (np.linalg.inv(np.array(self.sampler.cutter.tcp('tcp').tf,
                                     dtype=np.float64))
              @ np.array(self.sampler.tcp('flange').tf, dtype=np.float64)
              @ self.tf_flange_cam)
        return float(-tf[2, 3])

    def s_observe(self):
        start = self.qs_now()
        height = LOOK_CLEARANCE + self.eye_behind_tip(start)
        goal = lp.observe_goal(self.sampler, self.pln_ctx, self.tf_flange_cam,
                               self.eye, self.plant, start, height)
        path = None if goal is None else lp.plan(self.pln_ctx, start, goal)
        if path is None:
            raise RuntimeError('observe: no collision-free pose / path')
        return (f'to the observation pose: the nozzle '
                f'{LOOK_CLEARANCE * 1000:.0f} mm over the plant (the eye '
                f'{height * 1000:.0f} mm, {len(path)} waypoints)',
                lambda: self.run(path, FREE_SPEED, 'observe'))

    def leaf_row(self, i, lf, cells=False):
        """One leaf's numbers, one line: C, slope, ovalness, C - J, room,
        and why it is not taken (``cells``: as the live view's table row,
        J's height too)."""
        sl = leaf_pitch(lf)
        cj = np.linalg.norm(lf['C'] - lf['J'])
        why = []
        if cj < lv.MIN_CJ:
            why.append(f'C-J<{lv.MIN_CJ * 1000:.0f}')
        if min(lf['room']) < lv.MIN_ROOM:
            why.append(f'room<{lv.MIN_ROOM:.0f}')
        if any(np.linalg.norm(lf['C'][:2] - np.asarray(p)[:2])
               <= lv.SKIP_RADIUS for p in self.tried):
            why.append('tried')
        if cells:
            c = lf['C'] * 1000
            return [str(i), f'{c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}',
                    f"{lf['J'][2] * 1000:.0f}",
                    '?' if sl is None else f'{np.degrees(sl):+.0f}',
                    f"{lf.get('oval', float('nan')):.2f}", f'{cj * 1000:.1f}',
                    f"{min(lf['room']):.1f}", ' '.join(why)]
        return (f"leaf {i}: C {np.round(lf['C'] * 1000, 1)} mm, slope "
                f"{'?' if sl is None else f'{np.degrees(sl):+.0f}'} deg, "
                f"oval {lf.get('oval', float('nan')):.2f}, C-J "
                f"{cj * 1000:.1f} mm, room {min(lf['room']):.1f} mm"
                f"{' -- not taken: ' + ', '.join(why) if why else ''}")

    def publish_look(self, qs, title, leaves=(), chosen=None):
        """The last frame to the live view (``coripps.monitor``): the
        picture with ``leaves`` (``chosen`` starred) and their numbers, and
        its point cloud. Never fails the step."""
        try:
            eye = self.eye
            color, depth, K = eye.last_color, eye.last_depth, eye.last_K
            cam_tf = lp.tf_w_cam(self.sampler, qs, self.tf_flange_cam)
            ok = [lv.takeable(lf, self.tried) for lf in leaves]
            img = monitor.look_image(lv.normalize(color), leaves, chosen,
                                     takeable=lambda lf: lv.takeable(
                                         lf, self.tried))
            table = {
                'head': ['#', 'C x, y, z', 'J z', 'slope', 'oval', 'C-J',
                         'room', 'not taken'],
                'rows': [(self.leaf_row(i, lf, cells=True),
                          'chosen' if i == chosen else
                          'ok' if ok[i] else 'no')
                         for i, lf in enumerate(leaves)],
                'notes': ['mm, deg; taken: C-J >= %.0f, room >= %.0f '
                          '(oval shown only)' % (lv.MIN_CJ * 1000,
                                                 lv.MIN_ROOM),
                          'chosen: the highest J (within %.0f mm), then the '
                          'most room' % (lv.HEIGHT_BAND * 1000)]}
            monitor.publish(img, monitor.cloud_world(color, depth, K, cam_tf),
                            title, table)
        except Exception as exc:        # the view only -- go on
            print(f'(live view: {exc})')

    def s_look(self):
        def run():
            time.sleep(lp.SETTLE)
            qs = self.qs_now()
            found = lp.look(self.eye, self.sampler, qs,
                            self.tf_flange_cam, self.segmenter, self.plant,
                            skip=self.tried)
            if found is None:
                self.publish_look(qs, 'look: no leaves found')
                raise RuntimeError('look: no leaves found')
            leaves, chosen = found
            for i, lf in enumerate(leaves):
                print(f'  {self.leaf_row(i, lf)}')
            # a leaf asked for (``target``, world): the one nearest it,
            # whatever the rules -- unless it was tried already
            if self.target is not None and all(
                    np.linalg.norm(np.asarray(p)[:2] - self.target[:2])
                    > lv.SKIP_RADIUS for p in self.tried):
                chosen = int(np.argmin([np.linalg.norm(lf['C'][:2]
                                                       - self.target[:2])
                                        for lf in leaves]))
                print(f'look: the leaf asked for: leaf {chosen}, '
                      f"{np.linalg.norm(leaves[chosen]['C'][:2] - self.target[:2]) * 1000:.1f}"
                      ' mm from it (the rules not applied)')
            self.publish_look(qs, f'look: {len(leaves)} leaves, chosen '
                              f"{'none' if chosen is None else chosen}",
                              leaves, chosen)
            if chosen is None:
                raise RuntimeError(f'look: no leaf to take -- whole, C at '
                                   f'least {lv.MIN_CJ * 1000:.0f} mm from J, '
                                   f'{lv.MIN_ROOM:.0f} mm room at J, not '
                                   f'one of the {len(self.tried)} tried')
            self.leaf = leaves[chosen]
            self.leaves = leaves            # the others: s_close's fallback
            print(f"look: {len(leaves)} leaves, chosen leaf {chosen} -- C "
                  f"{np.round(self.leaf['C'] * 1000, 1)} mm, room at J "
                  f"{self.leaf['room'][0]:.1f} / {self.leaf['room'][1]:.1f} "
                  'mm')
        return 'look: the leaves and the one to take', run

    def s_close(self):
        """The eye CLOSE_LOOK_HEIGHT over the plant's top, the nozzle turned
        about the vertical as for the take (J -> C; by a fresh IK -- the
        wrist free to take the other way round, not turned where it is),
        shifted so the chosen leaf is clear of the cutter in the
        picture (from over it, at the approach, the nozzle hides it);
        straight there, or planned if not. No such pose / path for the
        chosen leaf: the next of this look's leaves (``lv.rank``: the
        highest, then the most room; the tried ones left out) -- not a
        failed take, nothing touched yet."""
        start = self.qs_now()
        rest = [self.leaves[k] for k in lv.rank(self.leaves, self.tried)
                if self.leaves[k] is not self.leaf]
        for leaf in [self.leaf] + rest:
            got = self.close_path(start, leaf)
            if got is not None:
                break
            print(f"close look: the leaf at C {np.round(leaf['C'] * 1000, 1)}"
                  ' mm not reachable -- the next one')
        else:
            raise TakeFailed('close look: no leaf of this look reachable')
        if leaf is not self.leaf:
            print(f"close look: taking the leaf at C "
                  f"{np.round(leaf['C'] * 1000, 1)} mm instead (room at J "
                  f"{leaf['room'][0]:.1f} / {leaf['room'][1]:.1f} mm)")
            self.leaf = leaf
        path, how = got
        return (f'to the close look: the eye {CLOSE_LOOK_HEIGHT * 1000:.0f} '
                f'mm over the plant\'s top, the leaf beside the cutter '
                f'({how})', lambda: self.run(path, NEAR_SPEED, 'close'))

    def close_path(self, start, leaf):
        """``(path, how)`` to the close look at ``leaf``, or None."""
        goal = lp.relook_goal(self.sampler, self.pln_ctx, self.tf_flange_cam,
                              self.eye, leaf, self.plant['plant_z'],
                              start, height=CLOSE_LOOK_HEIGHT)
        if goal is None:
            return None
        self.sampler.fk(qs=start)
        a = np.array(self.sampler.cutter.tcp('tcp').tf, dtype=np.float64)
        self.sampler.fk(qs=goal)
        b = np.array(self.sampler.cutter.tcp('tcp').tf, dtype=np.float64)
        turn = np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2,
                                 -1, 1))
        n = max(int(np.ceil(np.linalg.norm(b[:3, 3] - a[:3, 3])
                            / STRAIGHT_STEP)),
                int(np.ceil(turn / lp.TURN_STEP)), 1)
        path = lp.straight(self.sampler, self.pln_ctx, start,
                           [a[:3, 3] + (b[:3, 3] - a[:3, 3]) * k / n
                            for k in range(1, n + 1)],
                           np.array(oum.rotmat_slerp(a[:3, :3], b[:3, :3],
                                                     n + 1)[1:]), 'close')
        how = 'straight'
        if path is None:
            path, how = lp.plan(self.pln_ctx, start, goal), 'planned'
        return None if path is None else (path, how)

    def s_close_look(self):
        def run():
            time.sleep(lp.SETTLE)
            qs = self.qs_now()
            new = lp.relook(self.eye, self.sampler, qs,
                            self.tf_flange_cam, self.segmenter, self.leaf,
                            prefix='close_')
            self.publish_look(qs, 'close look: ' + (
                'the leaf not found again' if new is None else
                'the chosen leaf again'), [] if new is None else [new],
                None if new is None else 0)
            if new is None:
                raise TakeFailed('close look: the leaf not found again')
            self.leaf = new
            # its plane, and its whole outline on it (world): the descent
            # finds the leaf by it, however much of it the nozzle hides
            cam_tf = lp.tf_w_cam(self.sampler, self.qs_now(),
                                 self.tf_flange_cam)
            plane = leaf_plane(new)
            self.leaf_plane = (plane if plane is not None else
                               np.array([0.0, 0.0, new['C'][2]]))
            cnts, _ = cv2.findContours(new['mask'].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            K = self.eye.K
            px = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
            rays = np.c_[(px[:, 0] - K[0, 2]) / K[0, 0],
                         (px[:, 1] - K[1, 2]) / K[1, 1],
                         np.ones(len(px))] @ cam_tf[:3, :3].T
            self.leaf_outline = np.array([on_plane(cam_tf[:3, 3], r,
                                                   self.leaf_plane)
                                          for r in rays])
            # C on the plane too (where the nozzle is aimed, found again)
            self.leaf_c = on_plane(cam_tf[:3, 3], new['C'] - cam_tf[:3, 3],
                                   self.leaf_plane)
            # its slope J -> C from its plane: the nozzle comes in pitched
            # as much (at most PITCH_MAX_DEG; drooping, or unknown: level)
            slope = leaf_pitch(new)
            pitch = (float(np.clip(np.degrees(slope), 0.0, PITCH_MAX_DEG))
                     if slope is not None else 0.0)
            self.leaf['tf'] = pitched(new['tf'], np.deg2rad(pitch))
            print(f'close look: the leaf sloped '
                  f'{"?" if slope is None else f"{np.degrees(slope):+.0f}"} '
                  f'deg J -> C (+ up) -- the nozzle in '
                  f'{f"pitched {pitch:.0f} deg" if pitch else "level"}')
        return ('close look: the chosen leaf again, its frame the take goes '
                'by (no motion)', run)

    def s_onto_leaf(self):
        path = lp.to_leaf(self.sampler, self.pln_ctx, self.leaf['tf'],
                          self.qs_now())
        if path is None:                # pitched or level: not at a slant
            raise TakeFailed('onto the leaf: not plannable at its pitch '
                             f'({np.degrees(np.arcsin(self.leaf["tf"][2, 1])):.0f}'
                             ' deg) -- another leaf')
        def run():
            self.run(path, NEAR_SPEED, 'onto the leaf')
            self.lifted = False
        return (f'onto the leaf: over its C, then straight down to '
                f'{lp.APPROACH_OFFSET * 1000:.0f} mm over it', run)

    def s_seal(self):
        """Vacuum on, then down along the leaf's z a SEAL_STEP at a time
        until the pressure sensor reads the nozzle sealed -- the leaf's
        height is a stereo guess (mm off), the sensor finds it -- at most
        DESCEND_DIST; not sealed by then: the take failed.

        On the way, while the nozzle is over SERVO_UNTIL above the leaf's
        height, it is centred on the leaf by the left eye before each step
        (:meth:`servo_error`): moved sideways by SERVO_GAIN of what is off,
        SERVO_MAX in all (the eye sees the leaf under the nozzle down to
        ~3 mm; below that, blind)."""
        # planned whole only to know it is plannable; run step by step
        self.line(DESCEND_DIST * self.leaf['tf'][:3, 2], 'seal',
                  step=SEAL_STEP)
        start = self.qs_now()

        def run():
            off = np.max(np.abs(self.qs_now() - start))
            if off > lp.ARRIVAL_TOLERANCE:
                raise RuntimeError(f'seal: the arm moved since planning '
                                   f'({np.degrees(off):.2f} deg)')
            self.cobotta.enable(speed=SEAL_SPEED)
            shift = np.zeros(2)                 # the sideways so far (m)
            sucking = False
            errs = []                   # the leaf's C off the nozzle's
                                        # start line (m), per measure
            n = int(round(DESCEND_DIST / SEAL_STEP))
            for k in range(n + 1):
                qs = self.qs_now()
                self.sampler.fk(qs=qs)
                cur = np.array(self.sampler.cutter.tcp('tcp').tf,
                               dtype=np.float64)
                above = cur[2, 3] - self.leaf['C'][2]
                if not sucking and above <= VAC_ON_ABOVE + 1e-4:
                    # on only this near: from farther its draw lifts the
                    # leaf, swung about its petiole off the nozzle's line
                    self.cutter.vacuum('suck')
                    time.sleep(SEAL_WAIT)
                    sucking = True
                    print(f'seal: vacuum on, {above * 1000:.1f} mm over the '
                          'leaf')
                if sucking and self.cutter.sealed():
                    print(f'seal: sealed {k * SEAL_STEP * 1000:.1f} mm past '
                          f'{lp.APPROACH_OFFSET * 1000:.0f} mm over the leaf, '
                          f'moved {np.round(shift * 1000, 1)} mm sideways')
                    time.sleep(SUCK_WAIT)
                    return
                if k == n:
                    break
                side = np.zeros(2)
                if cur[2, 3] - self.leaf['C'][2] > SERVO_UNTIL:
                    err = self.servo_error(qs, cur, k)
                    if err is not None:
                        # the measures since the last move, mean of the
                        # last SERVO_AVG -- each taken against the nozzle
                        # where it was then, so less what it moved since
                        errs.append(err + shift)
                        err = np.mean(errs[-SERVO_AVG:], axis=0) - shift
                    if err is not None and np.linalg.norm(err) > SERVO_DEADBAND:
                        side = SERVO_GAIN * err
                        room = SERVO_MAX - np.linalg.norm(shift)
                        if np.linalg.norm(side) > room:
                            side *= max(room, 0.0) / np.linalg.norm(side)
                        shift += side
                pos = cur[:3, 3] + np.array([*side, 0.0]) \
                    + SEAL_STEP * self.leaf['tf'][:3, 2]
                step = lp.straight(self.sampler, self.pln_ctx, qs, [pos],
                                   cur[:3, :3], 'seal')
                if step is None:
                    raise TakeFailed('seal: the next step not plannable')
                self.cobotta.move_path([step[-1]])
                time.sleep(SEAL_WAIT)
            raise TakeFailed(f'seal: not sealed {DESCEND_DIST * 1000:.0f} '
                             f'mm past {lp.APPROACH_OFFSET * 1000:.0f} mm '
                             'over the leaf')
        return (f'vacuum on, down until sealed (at most '
                f'{DESCEND_DIST * 1000:.0f} mm, {SEAL_STEP * 1000:.1f} mm a '
                'step), centred on the leaf by the eye on the way', run)

    def nozzle_mask(self):
        """The pixels of the left picture the cutter itself covers (its
        mesh, not grown -- fixed to the eye, the same from any pose)."""
        if getattr(self, '_nozzle_mask', None) is None:
            self.sampler.fk(qs=self.qs_now())
            tf = np.linalg.inv(np.array(self.sampler.tcp('flange').tf,
                                        dtype=np.float64) @ self.tf_flange_cam) \
                @ np.array(self.sampler.cutter.tcp('tcp').parent_lnk.tf,
                           dtype=np.float64)
            geom = lp.ogl.load_geometry(lp.CUTTER_MESH)
            vs = np.asarray(geom.vs, np.float64) @ tf[:3, :3].T + tf[:3, 3]
            tris = vs[np.asarray(geom.fs)]
            tris = tris[(tris[:, :, 2] > lp.MIN_TOOL_Z).all(axis=1)]
            K = self.eye.K
            px = np.stack([K[0, 0] * tris[..., 0] / tris[..., 2] + K[0, 2],
                           K[1, 1] * tris[..., 1] / tris[..., 2] + K[1, 2]], -1)
            mask = np.zeros((self.eye.size[1], self.eye.size[0]), np.uint8)
            for tri in np.round(px).astype(np.int32):
                cv2.fillConvexPoly(mask, tri, 1)
            self._nozzle_mask = mask > 0
        return self._nozzle_mask

    def servo_error(self, qs, cur, k=0):
        """How far (m, world x y) the leaf's C is from where the nozzle
        comes down, seen by the left eye now. Not the middle of what is
        seen of it (the nozzle hides more of it each step, and that middle
        jumps): the close look's whole outline, projected where it should
        be, shifted to fit the green seen (both less the nozzle's pixels),
        and C shifted as much. None when less than SERVO_MIN_SEEN of the
        outline is past the nozzle, or it fits worse than SERVO_MIN_MATCH.
        The picture, marked, saved as ``seal_servo_<k>.png``."""
        img = lv.normalize(self.eye.rect.rectify(*self.eye.cam.frames(
            warn=False))[0])
        h, w = img.shape[:2]
        cam_tf = lp.tf_w_cam(self.sampler, qs, self.tf_flange_cam)
        K = self.eye.K
        nozzle = self.nozzle_mask()
        # where the nozzle's axis meets the leaf's plane (it may be pitched)
        touch = on_plane(cur[:3, 3], cur[:3, 2], self.leaf_plane)
        poly = np.array([lp.project(p, cam_tf, K) for p in self.leaf_outline])
        pred = np.zeros((h, w), np.uint8)
        cv2.fillPoly(pred, [np.round(poly).astype(np.int32)], 1)
        whole = int(pred.sum())
        seen = (pred.astype(bool) & ~nozzle).sum() / max(whole, 1)
        b, g, r = (img[..., i].astype(np.int16) for i in range(3))
        green = ((g > r + SERVO_GREEN) & (g > b + SERVO_GREEN)
                 & ~nozzle).astype(np.uint8)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        c_px = lp.project(self.leaf_c, cam_tf, K)
        t_px = lp.project(touch, cam_tf, K)
        out = img.copy()
        cv2.circle(out, tuple(int(t) for t in np.round(t_px)), 4,
                   (255, 0, 255), 1)                       # where it comes down
        shift_px, score = None, 0.0
        vv, uu = np.nonzero(pred)
        if seen >= SERVO_MIN_SEEN and len(uu):
            depth = cam_tf[:3, 2] @ (self.leaf_c - cam_tf[:3, 3])
            rad = max(int(np.ceil(SERVO_SEARCH / depth * K[0, 0])), 1)
            # the window: the outline's box grown by the search
            y0, y1 = max(vv.min() - rad, 0), min(vv.max() + 1 + rad, h)
            x0, x1 = max(uu.min() - rad, 0), min(uu.max() + 1 + rad, w)
            leaf = np.pad(pred[y0:y1, x0:x1].astype(bool), rad)
            # the green counted against it: only near it (not the next leaf)
            near = cv2.dilate(leaf.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            free = ~nozzle[y0:y1, x0:x1]
            seen_g = green[y0:y1, x0:x1].astype(bool)
            hh, ww = y1 - y0, x1 - x0
            # each shift: the outline moved, compared only where the nozzle
            # does not hide it THERE (IoU with the green seen)
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    at = leaf[rad - dy:rad - dy + hh, rad - dx:rad - dx + ww]
                    inter = np.count_nonzero(at & seen_g)
                    if not inter:
                        continue
                    by = near[rad - dy:rad - dy + hh, rad - dx:rad - dx + ww]
                    iou = inter / np.count_nonzero((at & free) | (seen_g & by))
                    if iou > score:
                        score, shift_px = iou, np.array([dx, dy], np.float64)
        if shift_px is None or score < SERVO_MIN_MATCH:
            cv2.polylines(out, [np.round(poly).astype(np.int32)], True,
                          (128, 128, 128), 1)
            cv2.imwrite(os.path.join(lp.OBSERVE_DIR, f'seal_servo_{k:02d}.png'),
                        out)
            print(f'seal: the leaf {seen:.0%} past the nozzle'
                  f'{f", matched {score:.2f}" if shift_px is not None else ""}'
                  ' -- straight on')
            return None
        cv2.polylines(out, [np.round(poly + shift_px).astype(np.int32)], True,
                      (0, 255, 0), 1)
        c_now_px = c_px + shift_px
        cv2.circle(out, tuple(int(t) for t in np.round(c_now_px)), 3,
                   (0, 0, 255), -1)
        cv2.imwrite(os.path.join(lp.OBSERVE_DIR, f'seal_servo_{k:02d}.png'), out)
        ray = cam_tf[:3, :3] @ np.array([(c_now_px[0] - K[0, 2]) / K[0, 0],
                                         (c_now_px[1] - K[1, 2]) / K[1, 1], 1.0])
        c_now = on_plane(cam_tf[:3, 3], ray, self.leaf_plane)
        err = c_now[:2] - touch[:2]
        print(f'seal: {(cur[2, 3] - self.leaf["C"][2]) * 1000:4.1f} mm over the '
              f'leaf, its C {np.round(err * 1000, 1)} mm off ({seen:.0%} '
              f'past the nozzle, matched {score:.2f})')
        return err

    def held(self, what):
        """TakeFailed unless the pressure sensor still reads the nozzle sealed
        -- not carried on to the tube; the run backs up and takes again."""
        if DESCEND_DIST > 0 and not self.cutter.sealed():
            raise TakeFailed(f'{what}: the leaf is lost (the nozzle no longer '
                           'sealed)')

    def drop_take(self):
        """After a failed take: ultrasonic off, blade in, vacuum off."""
        self.cutter.ultrasonic(False)
        self.cutter.retract()
        self.cutter.vacuum('off')

    def s_back_up(self):
        """Back up after a lost leaf, before looking again: LIFT straight
        up from the plant, as the lift (not when lifted already)."""
        if self.lifted:
            return 'back up: lifted already', lambda: None
        path, how = self.lift_path()
        return (f'back up {LIFT * 1000:.0f} mm straight up off the plant, '
                f'{how}', lambda: self.run(path, NEAR_SPEED, 'back up'))

    def s_tilt(self):
        """The held leaf tilted up to TILT_TO_DEG (from the pitch the
        nozzle came in at; already that steep: nothing) about the plant's
        centre: the nozzle swung about the horizontal axis through the
        plant's centre (at the nozzle's height) across the centre -> nozzle
        line, the nozzle going up -- C and J both lifted, the leaf and the
        blade's sweep (between the nozzle and J) off the soil and the pot's
        rim. Planned when its turn comes; not plannable: another leaf."""
        qs = self.qs_now()
        self.sampler.fk(qs=qs)
        cur = np.array(self.sampler.cutter.tcp('tcp').tf, dtype=np.float64)
        y = self.leaf['tf'][:3, 1]                  # J -> C (pitched)
        pivot = np.array([*self.plant['center'][:2], cur[2, 3]])
        out = cur[:3, 3] - pivot                    # centre -> nozzle
        out[2] = 0.0
        if np.linalg.norm(out) < 0.005:             # over the centre: J -> C
            out = np.array([y[0], y[1], 0.0])
        out /= np.linalg.norm(out)
        axis = np.cross([0.0, 0.0, 1.0], out)
        # the sign that takes the nozzle (and C) up
        sign = 1.0 if (oum.rotmat_from_axangle(axis, 0.1) @ out)[2] > 0 \
            else -1.0
        now = np.degrees(np.arcsin(np.clip(y[2] / np.linalg.norm(y), -1, 1)))
        tilt = TILT_TO_DEG - now
        if tilt < 0.5:
            return (f'tilt: the leaf at {now:.0f} deg already (at least '
                    f'{TILT_TO_DEG:.0f}) -- not tilted', lambda: None)
        n = max(int(np.ceil(np.deg2rad(tilt) / lp.TURN_STEP)), 1)
        positions, rots = [], []
        for k in range(1, n + 1):
            r = oum.rotmat_from_axangle(axis, sign * np.deg2rad(tilt) * k / n)
            positions.append(pivot + r @ (cur[:3, 3] - pivot))
            rots.append(r @ cur[:3, :3])
        path = lp.straight(self.sampler, self.pln_ctx, qs, positions,
                           np.array(rots), 'tilt')
        if path is None:
            raise TakeFailed(f'tilt: the leaf not tiltable {tilt:.0f} deg up '
                             'here')

        def run():
            self.run(path, TILT_SPEED, 'tilt')
            self.held('tilt')
        r = np.linalg.norm((cur[:3, 3] - pivot)[:2])
        return (f'tilt the leaf {tilt:.0f} deg up about the plant centre, '
                f'from {now:.0f} to {TILT_TO_DEG:.0f} deg ({r * 1000:.0f} mm '
                'from the nozzle), '
                f'at {TILT_SPEED:.0f}%', run)

    def s_us_on(self):
        def run():
            self.cutter.ultrasonic(True)
            time.sleep(US_SPINUP)
        return f'ultrasonic on, {US_SPINUP:.1f} s', run

    def s_cut(self):
        def run():
            # after each pass: the leaf still on the nozzle? (the blade can
            # push it off) -- not: no more passes, the take again
            reached = self.cutter.cut(go_on=self.cutter.sealed)
            self.held('the cut')
            travel = self.cutter.cfg.travel_deg
            if reached[-1] < travel - 1.0:      # stalled on the petiole
                raise TakeFailed(f'the cut: the blade stopped at '
                                 f'{reached[-1]:.1f} of {travel:.0f} deg -- '
                                 'not cut')
        return (f'deploy the blade: the cut -- its last {xce.CUT_SLOW_DEG:.0f} '
                f'deg slow, up to {xce.CUT_PASSES} passes, the leaf checked '
                'after each', run)

    def s_us_off(self):
        return 'ultrasonic off', lambda: self.cutter.ultrasonic(False)

    def s_retract(self):
        def run():
            self.cutter.retract()
            self.held('retract the blade')
        return 'retract the blade, the leaf still held', run

    def s_lift(self):
        path, how = self.lift_path()

        def run():
            self.run(path, NEAR_SPEED, 'lift')
            self.lifted = True
            self.held('lift')
        return (f'lift {LIFT * 1000:.0f} mm straight up, the leaf held: '
                f'{LIFT_CLEAR * 1000:.0f} mm at its tilt, {how}', run)

    def lift_path(self):
        """LIFT straight up from where the arm is: ``(path, how)``."""
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
        return clear + rest[1:], how

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

    # -- the window: what each step left ------------------------------------
    # the vision steps' annotated pictures (saved by leaf_pipeline)
    _SAVED = {'s_find_plant': 'plant_found.png', 's_look': 'leaves.png',
              's_close_look': 'close_leaves.png'}

    def view(self, make, title):
        """The picture after step ``make`` ran: a vision step's annotated
        one (the plant's circle; the leaves: C red, J blue, the chosen one
        starred), else the left eye now, rectified, with the plant's circle
        and the chosen leaf's C -> J drawn where they are."""
        saved = self._SAVED.get(make.__name__)
        if saved is not None:
            img = cv2.imread(os.path.join(lp.OBSERVE_DIR, saved))
        else:
            img = lv.normalize(self.eye.rect.rectify(*self.eye.cam.frames())[0])
            cam_tf = lp.tf_w_cam(self.sampler, self.qs_now(), self.tf_flange_cam)
            K = self.eye.K

            def px(pt):
                return tuple(int(t) for t in np.round(lp.project(pt, cam_tf, K)))
            if self.plant is not None:
                c = self.plant['center']
                z = cam_tf[:3, 2] @ (c - cam_tf[:3, 3])
                cv2.circle(img, px(c), int(round(
                    self.plant['radius'] / z * K[0, 0])), (0, 0, 255), 2)
            if self.leaf is not None:
                cv2.line(img, px(self.leaf['C']), px(self.leaf['J']),
                         (0, 255, 255), 2)
                cv2.circle(img, px(self.leaf['C']), 4, (0, 0, 255), -1)
                cv2.circle(img, px(self.leaf['J']), 4, (255, 0, 0), -1)
        cv2.putText(img, title[:110], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3)
        cv2.putText(img, title[:110], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        return img

    def steps(self, retake=False):
        """The step makers, in order (each planned when its turn comes);
        ``retake``: after a failed take -- back up, then from the
        observation pose on (the plant as found; its leaves looked at again,
        another one chosen)."""
        steps = ([self.s_back_up] if retake else
                 [self.s_search, self.s_find_plant])
        steps += [self.s_observe, self.s_look, self.s_close,
                  self.s_close_look, self.s_onto_leaf]
        if DESCEND_DIST > 0:
            # the take: only when it touches -- a dry run's blade would
            # swing APPROACH_OFFSET over the leaves, its vacuum pull them
            steps += [self.s_seal, self.s_tilt, self.s_us_on, self.s_cut, self.s_us_off,
                      self.s_retract]
        return steps + [self.s_lift, self.s_to_capper, self.s_into_tube,
                        self.s_blow, self.s_out_of_tube]


def take(session, continuous=False, idle=None):
    """The steps, one after another (a failed take: another leaf, MAX_TRIES
    in all); each after a 'g' -- or straight through, ``continuous`` --
    (``idle``: keys from the live view's windows, as ``_wait_for_g``'s).
    True when the leaf is in the tube."""
    def stopped():
        if _quit_pressed():
            return True
        while idle is not None:
            typed = idle()
            if typed is None:
                return False
            if typed in _QUIT_KEYS:
                return True

    try:
        makers, k, tries, given_up = session.steps(), 0, 1, False
        while k < len(makers):
            make = makers[k]
            k += 1
            try:                    # planned (a step may fail the take here)
                what, run = make()
                title = (f'{"giving up, " if given_up else ""}'
                         f'{f"try {tries}/{MAX_TRIES}, " if tries > 1 else ""}'
                         f'step {k}/{len(makers)}: {what}')
                if continuous:
                    print(f'\n{title}')
                    if stopped():
                        print('stopped')
                        return False
                elif not _wait_for_g(title, idle=idle):
                    print('stopped')
                    return False
                run()
            except TakeFailed as exc:
                session.drop_take()
                session.tried.append(session.leaf['C'])   # another one next
                if tries >= MAX_TRIES:         # back up off it, then stop
                    print(f'{exc} -- {tries} tries failed: given up, backing '
                          'up off the plant')
                    makers, k, given_up = [session.s_back_up], 0, True
                    continue
                tries += 1
                print(f'{exc} -- back up, look again, take another leaf '
                      f'(try {tries}/{MAX_TRIES})')
                makers, k = session.steps(retake=True), 0
                continue
            # the picture: the vision steps' (leaves, numbers, cloud) went to
            # the live view as they looked; after the others, the eye now
            if make.__name__ not in LeafSession._SAVED:
                try:
                    monitor.publish(session.view(make, ''), title=title)
                except Exception as exc:    # the picture only -- go on
                    print(f'(no picture: {exc})')
        if given_up:
            print(f'\ngiven up: no leaf after {MAX_TRIES} tries')
            return False
        print('\ndone: the leaf is in the tube')
        return True
    except (RuntimeError, OSError, BCapError) as exc:
        print(f'stopped: {exc}')
        return False


def main(continuous=False):
    """:func:`take` in a thread of its own, the live view
    (``coripps.monitor``: the arms, the last look's cloud and picture with
    every leaf's numbers) on this one; closing it stops before the next
    step."""
    import threading
    import one.viewer.world as ovw
    from coripps import single_routine as sr

    try:
        session = LeafSession()
    except Exception as exc:
        print(f'cannot start: {exc}')
        return False
    try:
        base = ovw.World(cam_pos=sr.CAM_POS, cam_lookat_pos=sr.CAM_LOOKAT)
        live = monitor.Live(base, sr._Stage(base), keys=True).start()
        result = {}

        def work():
            result['ok'] = take(session, continuous, idle=live.key)
            print('(close the viewer to end)')

        thread = threading.Thread(target=work, name='steps', daemon=True)
        thread.start()
        base.run()
        live.closed = True                  # 'q' to the steps
        thread.join()
        return result.get('ok', False)
    finally:
        session.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--c', action='store_true',
                        help='continuous: no g per step (q stops)')
    args = parser.parse_args()
    sys.exit(0 if main(continuous=args.c) else 1)
