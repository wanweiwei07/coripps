"""Step through one tube's pick, from above by its cap, one action per key.

Headless apart from the camera window. Each ``g`` runs exactly one step,
then waits::

    1  move to pre_pick -- above the cap, the hand at layout_pick.LOOK_ROTMAT
       (planned from wherever the arm is now)
    2  open the gripper for the camera (wider: the fingers out of its
       view), look at the cap (vision/picker_rgb/
       inrack_cap_orientation): its long tab's direction; plan the grip --
       the hand's x along the long tab-strap line, towards the long tab or
       away (the one J6 reaches, the nearer), the fingers closing across it
       -- and every move to it
    3  open the gripper to layout_pick.PICK_OPEN_WIDTH: just wider than the
       cap, the fingers clear of the neighbours' tabs
    4  turn the hand to the grip, on the spot (over the pick point: the
       grip goes where layout_pick puts the tube, the camera gives only the
       long tab's direction)
    5  straight down to the cap
    6  close the gripper
    7  straight up layout_pick.LIFT_DIST with the tube

``q`` quits at any point. The moves are straight TCP lines / turns on the
spot solved by IK at run time (the grip depends on the cap) and checked
before they are sent: the arm has to still be where the previous step left
it, and the move has to be collision-free. Any failure stops the run with
nothing further sent.

    py -3.12 coripps/tool/tube/pick_sequence_single.py
    py -3.12 coripps/tool/tube/pick_sequence_single.py --tube 5
    py -3.12 coripps/tool/tube/pick_sequence_single.py --no-grip
    py -3.12 coripps/tool/tube/pick_sequence_single.py --grip-current 60

* ``--no-grip`` skips step 6, so the lift runs with the fingers open.
* ``--grip-current N`` closes with ``goal_current`` N (2.69 mA per unit; the
  default is the gripper Config's).

:class:`PickSession` is the reusable part: other step-through scripts (e.g.
``pick_cap_store``) take its pick steps and append their own.
"""

import sys
import argparse
from dataclasses import replace

import cv2
import numpy as np

from one.drivers.cobotta import BCapError, CobottaMotion

import one.motion.core.constraint as omcc
import one.motion.core.planning_context as omppc

import coripps.picker_motion as pkm
from coripps.tool.keys import _quit_pressed, _wait_for_g
import coripps.tool.tube.layout_pick as layout
import coripps.tool.tube.layout_cap as cap
import coripps.tool.tube.poses_io_pick as ppio
import coripps.tool.tube.tcp_moves as otm
import coripps.vision.picker_rgb.inrack_cap_orientation as cor
from coripps.robots.mobisys import mobisys
from coripps.end_effectors.xh430_gripper.exec import (XH430GripperExec,
                                                      DynamixelError)


TUBE_INDEX = 2                          # which pick point to run
_EXEC_SPEED = 30.0                      # a step-by-step test: slow on purpose
_START_TOLERANCE = np.deg2rad(2.0)      # arm must still be where it was left
TURN_PLAY_EVERY = 10                    # step 4 plays the turn's 1-deg
                                        # samples this far apart
# a close counts as a grip only when the fingers stopped this far apart (a
# cap ~22 mm; closed on nothing they meet at ~4.5)
_GRIPPED_MM = 15.0
# a move counts as done only when the arm stopped this close to its end --
# else something stopped it (collision detection) and the run stops there
_ARRIVAL_TOLERANCE = np.deg2rad(1.0)


class MinTcpHeight(omcc.Constraint):
    """A planning constraint: ``tcp`` (of ``robot``) at or over ``min_z``
    (world)."""

    def __init__(self, robot, tcp, min_z):
        self.robot, self.tcp, self.min_z = robot, tcp, float(min_z)

    def is_valid(self, q):
        self.robot.fk(qs=q)
        if float(self.tcp.tf[2, 3]) < self.min_z:
            self.last_fail = f'tcp under {self.min_z * 1000:.0f} mm'
            return False
        self.last_fail = None
        return True


class PickSession:
    """Picks on the real picker: the planning context, the pick points, the
    camera's cap detection (SAM 3 and the cap template, loaded here), the
    arm and the gripper, plus the step verbs. One tube is selected at a time
    (:meth:`select`; ``tube`` None: none yet); the hardware stays connected
    across tubes.

    Raises ``RuntimeError`` when the tube cannot be run, the cap template is
    missing or the hardware is unreachable; nothing is sent before
    :meth:`run`.
    """

    def __init__(self, tube, grip_current=None):
        _, self.picker, sampler = mobisys()
        pkm.freeze_sampler(sampler)
        self.pln_ctx, _ = pkm.build_planning_context(self.picker, sampler)
        self.grip_current = grip_current
        self.points, self.look_qs_all, _ = ppio.load_checked(
            ndof=self.picker.ndof, layout=layout)
        cor.template()                  # none: stop before anything moves
        self.seg = cor.segmenter()
        self.moves = otm.TcpMoves(self)  # straight lines, turns on the spot
        self._height_ctx = {}           # min_z -> its planning context
        self.grip = None                # step 2's plan
        self.quit = False               # set when 'q' ended a run
        if tube is not None:
            self.select(tube)

        try:
            self.cobotta = CobottaMotion(pkm._PICKER_HOST,
                                         speed=_EXEC_SPEED).connect()
        except (OSError, BCapError) as exc:
            raise RuntimeError(f'picker: cannot reach {pkm._PICKER_HOST} '
                               f'({exc})') from exc
        try:
            self.gripper = XH430GripperExec().connect()
        except DynamixelError as exc:
            self.cobotta.disconnect()
            raise RuntimeError(f'gripper: {exc}') from exc

    def select(self, tube, verbose=True):
        """Make ``tube`` the one the step verbs act on. Raises
        ``RuntimeError`` when it is not in the file or its pre_pick is in
        collision."""
        if not 0 <= tube < len(self.points):
            raise RuntimeError(f'tube {tube} of {len(self.points)} pick '
                               'points')
        self.tube = tube
        self.point = self.points[tube]
        self.look_qs = self.look_qs_all[tube]
        self.grip = None
        if verbose:
            print(f'tube {tube}: grasp at {np.round(self.point * 1000, 1)} '
                  f'mm, pre_pick {np.round(np.degrees(self.look_qs), 2)} deg')
        if not self.pln_ctx.is_state_valid(self.look_qs):
            raise RuntimeError(f'tube {tube}: its pre_pick is in collision')

    def release(self):
        """The arm: motors off, given back. The gripper: port only, so a held
        tube stays held."""
        self.cobotta.disconnect()
        self.gripper.disconnect()

    def real_qs(self):
        return self.pln_ctx.enforce_bounds(
            self.cobotta.joint_angles(ndof=self.picker.ndof))

    # ---- the grip -----------------------------------------------------------
    def look(self):
        """The cap under the camera: its long tab's direction (deg, world),
        None when not found."""
        res = cor.detect(cor.grab(), cor.camera_tf(self.picker,
                                                   self.real_qs()),
                         self.point[:2], seg=self.seg)
        print(f'  {cor.report(res)}')
        cv2.imshow(cor.WINDOW, cor.draw(res))
        cv2.waitKey(1)
        return res['long_deg']

    def plan_grip(self, tab_deg, flips=(False, True)):
        """From pre_pick (where the arm stands): the hand turned on the spot
        to the grip yaw, straight down to the cap, straight up LIFT_DIST with
        it -- for the grip yaw towards the long tab or away from it, the
        smaller turn first (``flips``: only these). :attr:`grip` (``flip``:
        away), or None when none can be."""
        m = self.moves
        start = self.real_qs()
        _, look_rotmat = m.tcp_pose(start)
        options = []
        for flip in flips:
            target = layout.grip_rotmat(tab_deg, flip)
            turn = otm.wrap180(np.degrees(
                np.arctan2(target[1, 0], target[0, 0])
                - np.arctan2(look_rotmat[1, 0], look_rotmat[0, 0])))
            options.append((abs(turn), flip, turn))
        for _, flip, turn in sorted(options):
            path, _ = m.arc_path(start, turn)
            down = None if path is None else m.line_path(
                path[-1], [0, 0, -layout.PICK_LOOK_RISE])
            up = None if down is None else m.line_path(
                down[-1], [0, 0, layout.LIFT_DIST])
            if up is not None:
                # played every TURN_PLAY_EVERY-th 1-deg sample: each sample
                # is a move of its own, and 1-deg moves never get up to speed
                # (the hand empty, 10 mm over the caps; a 10-deg stretch
                # strays from the turn by < 1 mm)
                turn_play = list(path[::TURN_PLAY_EVERY])
                if not np.array_equal(turn_play[-1], path[-1]):
                    turn_play.append(path[-1])
                self.grip = {'tab': tab_deg, 'flip': flip, 'turn': turn_play,
                             'turn_deg': turn, 'down': down, 'up': up}
                return self.grip
        return None

    def look_and_plan(self):
        """Step 2: open for the camera, look, plan the grip. False when the
        look or the plan fails."""
        self.grip = None
        self.work_gripper('camera')
        tab = self.look()
        if tab is None:
            print('  the long tab is not found -- stopping')
            return False
        grip = self.plan_grip(tab)
        if grip is None:
            print(f'  long tab {tab:.0f} deg: no grip reachable -- stopping')
            return False
        print(f'  long tab {tab:.0f} deg: turn the hand '
              f'{grip["turn_deg"]:+.0f} deg, its x '
              f'{"away from" if grip["flip"] else "towards"} the long tab')
        return True

    # ---- step verbs: each returns True to go on, False to stop -------------
    def transfer_to(self, name, goal, speed=None, held=False):
        """From wherever the arm is to ``goal``, nothing under
        ``layout_cap.TRAVEL_MIN_Z`` on the way -- ``held`` (a tube in the
        gripper): grasp_center over HELD_TRAVEL_MIN_Z, the tube's bottom over
        TRAVEL_MIN_Z: straight up to it first when the arm is below, across
        by RRT-Connect (``one``) with that height one of its validity
        constraints (the straight joint line first, when it is free),
        straight down to ``goal`` last when it is below.
        ``speed`` (percent) overrides the run speed for these moves only."""
        start = self.real_qs()
        goal = np.asarray(goal, dtype=np.float32)
        if (np.abs(start - goal).max() < _START_TOLERANCE
                and self.pln_ctx.is_motion_valid(start, goal)):
            # (nearly) there already: straight on, no up-and-down on the spot
            return self.play(f'{name} (nearly there)', [start, goal],
                             speed=speed)
        if not self.pln_ctx.is_state_valid(start):
            print('  the current pose is in collision -- cannot plan from it')
            return False
        min_z = cap.HELD_TRAVEL_MIN_Z if held else cap.TRAVEL_MIN_Z
        plan = self.plan_transfer(start, np.asarray(goal, dtype=np.float32),
                                  min_z)
        if plan is None:
            print(f'  {name}: no path from here with grasp_center at or over '
                  f'{min_z * 1000:.0f} mm -- stopping')
            return False
        # one continuous path: played part by part, the arm stops at the end
        # of each (a jerk, worst where a part is a few mm or a fraction of a
        # degree long); an across too short to matter is left out
        waypoints, names = [plan[0][1][0]], []
        for part, path in plan:
            if (part == 'across' and np.abs(np.asarray(path[-1])
                                            - waypoints[-1]).max()
                    < _START_TOLERANCE):
                continue
            waypoints += list(path[1:])
            names.append(part)
        return self.play(f"{name} ({' + '.join(names)})", waypoints,
                         speed=speed)

    def plan_transfer(self, start, goal, min_z=cap.TRAVEL_MIN_Z):
        """``[(part, waypoints), ...]`` from joints ``start`` to ``goal``:
        'up' / 'across' / 'down' as :meth:`transfer_to` describes, nothing
        under ``min_z``; None when it cannot be."""
        m = self.moves
        s_pos, _ = m.tcp_pose(start)
        g_pos, g_rot = m.tcp_pose(goal)
        parts, a, b, down = [], start, goal, None
        if s_pos[2] < min_z:
            up = m.line_path(start, [0, 0, min_z - s_pos[2]])
            if up is None:
                # no straight line up (e.g. the wrist near its singularity):
                # plan to the same pose at min_z, collisions the only check
                _, s_rot = m.tcp_pose(start)
                sols = [q for q in self.picker.ik(
                            np.array([s_pos[0], s_pos[1], min_z]), s_rot,
                            tcp=m.tcp, ref_qs=np.array(start))
                        if self.pln_ctx.is_state_valid(q)]
                if not sols:
                    return None
                print('  up: no straight line -- planning it instead')
                top = min(sols, key=lambda q: np.abs(q - start).max()).astype(
                    np.float32)
                up = pkm.build_transfer(self.pln_ctx, start, top,
                                        force_plan=False)
            if up is None:
                return None
            parts.append(('up', up))
            a = up[-1]
        if g_pos[2] < min_z:
            sols = [q for q in self.picker.ik(
                        np.array([g_pos[0], g_pos[1], min_z]), g_rot,
                        tcp=m.tcp, ref_qs=np.array(goal))
                    if self.pln_ctx.is_state_valid(q)]
            if not sols:
                return None
            b = min(sols, key=lambda q: np.abs(q - goal).max()).astype(
                np.float32)
            down = m.line_path(b, [0, 0, g_pos[2] - min_z])
            if down is None:
                return None
        # the straight joint line when it is free (the height included),
        # RRT-Connect only when it is not
        across = pkm.build_transfer(self.height_ctx(min_z), a, b,
                                    force_plan=False)
        if across is None:
            return None
        parts.append(('across', across))
        if down is not None:
            parts.append(('down', down))
        return parts

    def height_ctx(self, min_z):
        """The planning context (the same collider) with grasp_center at or
        over ``min_z`` one of its validity constraints (1 mm slack: the
        ends of the across are AT it)."""
        if min_z not in self._height_ctx:
            self._height_ctx[min_z] = omppc.PlanningContext(
                collider=self.pln_ctx.collider,
                cd_step_size=self.pln_ctx.cd_step_size,
                constraints=(MinTcpHeight(self.picker, self.moves.tcp,
                                          min_z - 0.001),))
        return self._height_ctx[min_z]

    def follow_path(self, name, path, speed=None):
        """Play a joint path whose first config is where the arm should
        already be standing. ``speed`` (percent) overrides the run speed for
        this path only."""
        start = self.real_qs()
        drift = np.max(np.abs(start - path[0]))
        if drift > _START_TOLERANCE:
            print(f'  the arm moved since the last step '
                  f'({np.degrees(drift):.2f} deg) -- stopping')
            return False
        segment = [start] + list(path[1:])
        if not pkm.local_sequence_valid(self.pln_ctx, segment):
            print(f'  {name}: the move is in collision -- stopping')
            return False
        return self.play(name, segment, speed=speed)

    def play(self, name, waypoints, speed=None):
        """Run ``waypoints``; ``speed`` (percent) for this move only, the run
        speed is put back afterwards even when the move fails."""
        self.cobotta.enable(speed=speed)     # a no-op once already powered
        try:
            self.cobotta.move_path(waypoints)  # passes through, stops at end
        finally:
            if speed is not None:
                try:
                    self.cobotta.set_speed(_EXEC_SPEED)
                except (OSError, BCapError):
                    # refused after a stop, but set_speed has already stored
                    # the run speed, so the next enable() re-applies it
                    pass
        reached = self.real_qs()
        print(f'  at {name} ({len(waypoints) - 1} moves): '
              f'{np.round(np.degrees(reached), 2)} deg')
        off = np.max(np.abs(reached - np.asarray(waypoints[-1])))
        if off > _ARRIVAL_TOLERANCE:
            print(f'  {name}: stopped {np.degrees(off):.2f} deg short of its '
                  'end (hit something?) -- stopping')
            return False
        return True

    def work_gripper(self, verb):
        gripper = self.gripper
        if verb == 'open':
            gripper.open()
        elif verb == 'camera':          # open for the camera
            gripper.open_for_camera()
        elif verb == 'pick':            # just wider than the cap
            gripper.move_to_mm(layout.PICK_OPEN_WIDTH * 1000)
        else:
            if self.grip_current is not None:
                gripper.cfg = replace(gripper.cfg,
                                      goal_current=self.grip_current)
            gripper.close()
        print(f'  gripper {verb}: {gripper.angle_deg():5.2f} deg / '
              f'{gripper.opening_mm():5.2f} mm @ '
              f'{gripper.dxl.present_current_ma():4.0f} mA')
        return True

    def grip_tube(self):
        """Close on a tube; False (the gripper left closed) when the fingers
        met nothing."""
        self.work_gripper('close')
        if self.gripper.opening_mm() < _GRIPPED_MM:
            print(f'  closed to {self.gripper.opening_mm():.1f} mm: no tube '
                  'between the fingers -- stopping')
            return False
        return True

    def pick_steps(self, no_grip=False, free_speed=None, near_speed=None):
        """The seven steps from wherever the arm is to the lift, tube held.

        ``free_speed`` is for the transfer to pre_pick (nothing touched),
        ``near_speed`` for the turn, the descent onto the cap and the lift;
        None keeps the run speed.
        """
        g = lambda key: self.grip[key]
        return [
            ('move to pre_pick (look from above)',
             lambda: self.transfer_to('pre_pick', self.look_qs,
                                      speed=free_speed)),
            ('open for the camera, look at the cap, plan the grip',
             self.look_and_plan),
            (f'open the gripper to {layout.PICK_OPEN_WIDTH * 1000:.0f} mm '
             '(clear of the neighbours)', lambda: self.work_gripper('pick')),
            ('turn the hand to the grip',
             lambda: self.follow_path('the grip yaw', g('turn'),
                                      speed=near_speed)),
            (f'down {layout.PICK_LOOK_RISE * 1000:.0f} mm to the cap',
             lambda: self.follow_path('pick', g('down'), speed=near_speed)),
            ('close the gripper on the cap', self.grip_tube)
            if not no_grip else
            ('SKIP closing the gripper (--no-grip)', lambda: True),
            (f'lift {layout.LIFT_DIST * 1000:.0f} mm',
             lambda: self.follow_path('lift', g('up'), speed=near_speed)),
        ]

    def run(self, steps, release=True, title=None, continuous=False):
        """Wait for 'g' before each step -- or, ``continuous``, run them
        back to back, still stopping before the next step when q / Esc /
        Ctrl+C has been pressed. True when every step ran or 'q' ended it
        (:attr:`quit` tells which), False on a failure. Releases the
        hardware at the end unless ``release`` is False -- a caller running
        several tubes releases once, itself."""
        print(f'picker at {np.round(np.degrees(self.real_qs()), 2)} deg, '
              f'running at {_EXEC_SPEED:.0f}%')
        prefix = '' if title is None else f'{title}, '
        ok = True
        self.quit = False
        try:
            for k, (label, step) in enumerate(steps):
                prompt = f'{prefix}step {k + 1}/{len(steps)}: {label}'
                if continuous:
                    go = not _quit_pressed()
                    if go:
                        print(f'\n{prompt}', flush=True)
                else:
                    go = _wait_for_g(prompt)
                if not go:
                    print('quit')
                    self.quit = True
                    break
                if not step():
                    ok = False
                    break
            else:
                print('\nall steps done')
        except (OSError, BCapError, DynamixelError) as exc:
            print(f'  failed: {exc} -- stopping')
            ok = False
        finally:
            if release:
                self.release()
        return ok


def main(tube=TUBE_INDEX, no_grip=False, grip_current=None):
    try:
        session = PickSession(tube, grip_current=grip_current)
    except RuntimeError as exc:
        print(exc)
        return False
    if no_grip:
        print('--no-grip: the gripper stays open, the lift touches nothing')
    elif grip_current is not None:
        print(f'--grip-current: closing at {grip_current} '
              f'(~{grip_current * 2.69:.0f} mA)')
    try:
        return session.run(session.pick_steps(no_grip=no_grip))
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tube', type=int, default=TUBE_INDEX,
                        help=f'pick point index (default {TUBE_INDEX})')
    parser.add_argument('--no-grip', action='store_true',
                        help='skip closing the gripper (lift with it open)')
    parser.add_argument('--grip-current', type=int, default=None,
                        help='goal_current for the close, 2.69 mA per unit')
    args = parser.parse_args()
    sys.exit(0 if main(tube=args.tube, no_grip=args.no_grip,
                       grip_current=args.grip_current) else 1)
