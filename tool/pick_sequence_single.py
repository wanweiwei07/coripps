"""Step through one precomputed tube's pick, one action per key press.

Headless: no viewer. Each ``g`` runs exactly one step, then waits::

    1  move to pre_pick_high   (planned from wherever the arm is now)
    2  open the gripper
    3  move to pre_pick
    4  move to pick
    5  close the gripper
    6  move to lift

``q`` quits at any point. Steps 3, 4 and 6 play the precomputed
straight-line samples of their segment, so the TCP moves in a straight line
(the lift no longer bows sideways against the rack). Every arm step is checked
before it is sent: the arm has to still be where the previous step left it,
and the move has to be collision-free. Any failure stops the run with nothing
further sent.

    py -3.12 coripps/tool/pick_sequence_single.py
    py -3.12 coripps/tool/pick_sequence_single.py --tube 5
    py -3.12 coripps/tool/pick_sequence_single.py --no-grip
    py -3.12 coripps/tool/pick_sequence_single.py --grip-current 60

Diagnosing a collision stop at the start of the lift:

* ``--no-grip`` skips step 5, so the lift runs with the blades open around
  the tube, touching nothing. Still tripping means the tube is not the cause.
* ``--grip-current N`` closes with ``goal_current`` N (2.69 mA per unit; the
  default is the gripper Config's) -- a lighter grip pulls the tube, and so the
  wrist, sideways less when the grasp is off-centre.

:class:`PickSession` is the reusable part: other step-through scripts (e.g.
``pick_cap_store``) take its six pick steps and append their own.
"""

import sys
import argparse
from dataclasses import replace

import numpy as np

from one.drivers.cobotta import BCapError, CobottaMotion

import coripps.picker_motion as pkm
import coripps.tool.layout_pick as layout
import coripps.tool.poses_io_pick as ppio
from coripps.robots.mobisys import mobisys
from coripps.end_effectors.xh430_gripper.exec import (XH430GripperExec,
                                                      DynamixelError)


TUBE_INDEX = 2                          # which precomputed pick to run
_EXEC_SPEED = 30.0                      # a step-by-step test: slow on purpose
_START_TOLERANCE = np.deg2rad(2.0)      # arm must still be where it was left

_QUIT_KEYS = ('q', '\x1b', '\x03')     # q, Esc, Ctrl+C

try:
    import msvcrt

    def _read_key():
        return msvcrt.getwch().lower()

    def _quit_pressed():
        """Non-blocking: True when q / Esc / Ctrl+C is waiting."""
        while msvcrt.kbhit():
            if msvcrt.getwch().lower() in _QUIT_KEYS:
                return True
        return False
except ImportError:                     # not Windows: fall back to a line
    def _read_key():
        return input().strip().lower()[:1]

    def _quit_pressed():
        return False                    # Ctrl+C still raises there


def _wait_for_g(prompt):
    """True on 'g', False on 'q' / Esc / Ctrl+C. Anything else is ignored.

    ``msvcrt.getwch`` hands Ctrl+C over as a plain '\\x03' instead of raising
    KeyboardInterrupt, so it is caught here explicitly.
    """
    print(f"\n{prompt} -- 'g' to run, 'q' to quit", flush=True)
    while True:
        key = _read_key()
        if key == 'g':
            return True
        if key in _QUIT_KEYS:
            return False


class PickSession:
    """Picks on the real picker: the planning context, the precomputed paths,
    the arm and the gripper, plus the step verbs. One tube is selected at a
    time (:meth:`select`); the hardware stays connected across tubes.

    Raises ``RuntimeError`` when the tube cannot be run or the hardware is
    unreachable; nothing is sent before :meth:`run`.
    """

    def __init__(self, tube, grip_current=None):
        _, self.picker, sampler = mobisys()
        pkm.freeze_sampler(sampler)
        self.pln_ctx, _ = pkm.build_planning_context(self.picker, sampler)
        self.grip_current = grip_current

        self.points, self.paths, self.rotmats, meta = ppio.load_checked(
            ndof=self.picker.ndof, pick_z=layout.PICK_Z)
        self.marks = [int(m) for m in meta['marks']]
        self.quit = False               # set when 'q' ended a run
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
            self.cobotta.close()
            raise RuntimeError(f'gripper: {exc}') from exc

    def select(self, tube, verbose=True):
        """Make ``tube`` the one the step verbs act on. Raises
        ``RuntimeError`` when it is not in the file or its own approach /
        lift is in collision."""
        if not 0 <= tube < len(self.points):
            raise RuntimeError(f'tube {tube} of {len(self.points)} '
                               'precomputed picks')
        self.tube = tube
        self.point, self.path = self.points[tube], self.paths[tube]
        self.rotmat = self.rotmats[tube]
        if verbose:
            print(f'tube {tube}: point={np.round(self.point, 4)}, '
                  f'{len(self.path)} path samples')
            for name, m in zip(ppio.POSE_NAMES, self.marks):
                print(f'  {name:14s} '
                      f'{np.round(np.degrees(self.path[m]), 2)} deg')
        if not pkm.local_sequence_valid(self.pln_ctx, self.path):
            raise RuntimeError(f'tube {tube}: the approach / lift is in '
                               'collision')

    def release(self):
        """The arm: motors off, given back. The gripper: port only, so a held
        tube stays held."""
        self.cobotta.close()
        self.gripper.close()

    def real_qs(self):
        return self.pln_ctx.enforce_bounds(
            self.cobotta.joint_angles(ndof=self.picker.ndof))

    # ---- step verbs: each returns True to go on, False to stop -------------
    def transfer_to(self, name, goal, speed=None):
        """Plan from wherever the arm is to ``goal`` (direct or RRT).
        ``speed`` (percent) overrides the run speed for this move only."""
        start = self.real_qs()
        if not self.pln_ctx.is_state_valid(start):
            print('  the current pose is in collision -- cannot plan from it')
            return False
        waypoints = pkm.build_transfer(self.pln_ctx, start, goal,
                                       force_plan=False)
        if waypoints is None:
            print(f'  {name}: no path from here -- stopping')
            return False
        return self.play(name, waypoints, speed=speed)

    def follow(self, name, m_from, m_to, speed=None):
        """One straight-line segment of the precomputed pick path."""
        return self.follow_path(name, self.path[m_from:m_to + 1], speed=speed)

    def follow_path(self, name, path, speed=None):
        """Play a precomputed joint path whose first config is where the arm
        should already be standing. ``speed`` (percent) overrides the run
        speed for this path only."""
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
        print(f'  at {name} ({len(waypoints) - 1} moves): '
              f'{np.round(np.degrees(self.real_qs()), 2)} deg')
        return True

    def work_gripper(self, verb):
        gripper = self.gripper
        if verb == 'open':
            gripper.open()
        else:
            if self.grip_current is not None:
                gripper.cfg = replace(gripper.cfg,
                                      goal_current=self.grip_current)
            gripper.close_jaws()
        print(f'  gripper {verb}: {gripper.angle_deg():5.2f} deg / '
              f'{gripper.opening_mm():5.2f} mm @ '
              f'{gripper.dxl.present_current_ma():4.0f} mA')
        return True

    def pick_steps(self, no_grip=False, free_speed=None, near_speed=None):
        """The six steps from wherever the arm is to ``lift``, tube held.

        ``free_speed`` is for the transfer to pre_pick_high (nothing touched),
        ``near_speed`` for the descent into the tube and the lift; None keeps
        the run speed.
        """
        m_high, m_pre, m_pick, m_lift = self.marks
        return [
            # step 1
            ('move to pre_pick_high',
             lambda: self.transfer_to('pre_pick_high', self.path[m_high],
                                      speed=free_speed)),
            # step 2
            ('open the gripper', lambda: self.work_gripper('open')),
            # step 3
            ('move to pre_pick',
             lambda: self.follow('pre_pick', m_high, m_pre,
                                 speed=near_speed)),
            # step 4
            ('move to pick',
             lambda: self.follow('pick', m_pre, m_pick, speed=near_speed)),
            # step 5
            ('close the gripper', lambda: self.work_gripper('close'))
            if not no_grip else
            ('SKIP closing the gripper (--no-grip)', lambda: True),
            # step 6
            ('move to lift',
             lambda: self.follow('lift', m_pick, m_lift, speed=near_speed)),
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
    return session.run(session.pick_steps(no_grip=no_grip))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tube', type=int, default=TUBE_INDEX,
                        help=f'precomputed pick index (default {TUBE_INDEX})')
    parser.add_argument('--no-grip', action='store_true',
                        help='skip closing the gripper (lift with it open)')
    parser.add_argument('--grip-current', type=int, default=None,
                        help='goal_current for the close, 2.69 mA per unit')
    args = parser.parse_args()
    sys.exit(0 if main(tube=args.tube, no_grip=args.no_grip,
                       grip_current=args.grip_current) else 1)
