"""Pick one tube, cap it in the store, pop it out and discard it, step by step.

Headless: no viewer. Each ``g`` runs exactly one step, then waits::

    1-6  the pick of ``pick_sequence_single`` (... -> close -> lift)
    7    move to the drop point, the tube still held
    8    lower straight down by ``layout_cap.PLACE_DESCEND``, still holding the
         tube -- it goes into the store instead of being dropped from the air
    9    open the gripper -- let go of the placed tube
    10   slide ``layout_cap.RELEASE_SHIFT`` along the TCP's own +x (the
         open blades leave the tube sideways), then rise straight up by
         ``layout_cap.RELEASE_RISE``
    11   close the gripper again
    12   move to the push start ``layout_cap.PUSH_POS`` -- over the back of
         the tube, just above it -- in ``layout_cap.PUSH_ROTMAT``
    13   move straight down by ``layout_cap.PUSH_DESCEND`` at
         :data:`PUSH_SPEED` -- the closed blades push the placed tube home
    14   retract straight back up to the push start (the push, reversed) --
         the store and the tube are not in the collision model, so the
         blades leave the way they came in before moving sideways
    15   move to the pop position ``layout_cap.POP_POS`` (the lever that
         pries the tube back out of the capper), in
         ``layout_cap.POP_ROTMAT``.
         MEANWHILE the capper opens the cap, presses it shut again and moves
         to its stopper position; the step ends when both are done
    16   press straight down by ``layout_cap.POP_DIST`` at :data:`POP_SPEED`
         on the lever, prying the tube out of the capper
    17   retract straight back up from the lever (the press, reversed)
    18   move the capper back home, ready for the next tube
    19   open the gripper
    20   move to the re-pick point ``layout_cap.REPICK_POS`` in
         ``layout_cap.REPICK_ROTMAT``
    21   lower straight down by ``layout_cap.REPICK_DESCEND`` to the popped
         tube, the gripper open
    22   close the gripper on the popped tube
    23   retract straight back up to the re-pick point, the tube held
    24   carry it to the discard point ``layout_cap.DISCARD_POS``
    25   open the gripper -- the tube drops into the bin
    26   move on: to the NEXT tube's pre_pick_high when there is one, else to
         the picker's home pose (``home_picker``)

Execution only: every pose and path comes from ``config/poses_cap.json``,
solved by ``precompute_poses_cap.py`` from ``layout_cap`` (the positions and
distances live there). Only the PLACING (steps 7-10) follows the tube's
own pick orientation; the push, the pop, the re-pick of the popped tube and
the discard are FIXED poses of the cell, the same for every tube. Before
anything moves the loaded paths are re-checked against the REAL sampler pose
-- no IK at run time. The tube, the store and the lever are not in the
collision model.

The capper runs in its own thread: the arm is a b-CAP socket, the capper a
separate serial port, so the two share nothing.

Speed by kind of move (percent, :data:`FREE_SPEED` etc.): free moves that
touch and carry nothing run fast; anything near or in contact with a tube,
or carrying one, runs slower -- a fast contact is a torque spike the
collision detection stops on, a fast carry can shake the tube loose.

    py -3.12 coripps/tool/pick_cap_store.py
    py -3.12 coripps/tool/pick_cap_store.py --tube 5
"""

import sys
import argparse
import threading

import numpy as np

import coripps.picker_motion as pkm
import coripps.tool.layout_cap as cap
import coripps.tool.poses_io_cap as cpio
from coripps.tool.home_picker import HOME_QS_DEG
from coripps.tool.pick_sequence_single import PickSession
from coripps.end_effectors.xh430_capper.exec import (XH430CapperExec,
                                                     DynamixelError)


TUBE_INDEX = 0
# Speeds (percent) by kind of move:
FREE_SPEED = 100.0   # steps 1, 14, 15, 26: nothing touched or carried
NEAR_SPEED = 100.0    # steps 3, 4, 6, 8, 10, 12, 17, 20, 21, 23: into / out
                     # of a tube, the lift, placing and leaving it, the push
                     # start
CARRY_SPEED = 100.0   # steps 7, 24: carrying a tube
PUSH_SPEED = 3.0     # step 13: a slow push keeps the contact torque under the
                     # collision detection threshold
POP_SPEED = 5.0      # step 16: pressing the lever, in contact as well


def load_cap_poses(session):
    """``poses_cap.json``, refused when missing, old or stale."""
    try:
        orientations, fixed, _meta = cpio.load_checked(
            ndof=session.picker.ndof, inputs=cap.inputs())
    except (OSError, ValueError) as exc:
        print(exc)
        return None
    return orientations, fixed


def pop_and_cap(session, capper, pop):
    """Step 15: the arm goes to the pop position while the capper opens the
    cap, presses it shut and goes to the stopper. True only when both
    succeed."""
    capper_result = {}

    def run_capper():
        try:
            capper.set_opening('open')
            capper_result['open'] = capper.servo_deg()
            capper.set_opening('close')
            capper_result['close'] = capper.servo_deg()
            capper_result['press_ma'] = capper.dxl.present_current_ma()
            capper.set_opening('stopper')
            capper_result['stopper'] = capper.servo_deg()
        except DynamixelError as exc:
            capper_result['error'] = exc

    worker = threading.Thread(target=run_capper, name='capper')
    worker.start()
    try:
        arm_ok = session.transfer_to('the pop position', pop,
                                     speed=FREE_SPEED)
    finally:
        worker.join()                   # never leave the capper mid-move
    if 'error' in capper_result:
        print(f"  capper failed: {capper_result['error']}")
        return False
    print(f"  capper opened to {capper_result['open']:6.2f} deg, pressed "
          f"shut at {capper_result['close']:6.2f} deg @ "
          f"{capper_result['press_ma']:4.0f} mA, at the stopper "
          f"{capper_result['stopper']:6.2f} deg")
    return arm_ok


def capper_home(capper):
    """Step 18: the capper back to its home angle."""
    capper.set_opening('home')
    print(f'  capper home at {capper.servo_deg():6.2f} deg')
    return True


def solve_after(session, next_tube):
    """Where the arm goes once the tube is discarded: ``(label, qs)`` --
    ``next_tube``'s pre_pick_high, or home when it is None. None when that
    pose is itself in collision."""
    if next_tube is not None:
        label = f'tube {next_tube} pre_pick_high'
        goal = session.paths[next_tube][session.marks[0]]
    else:
        label = 'home'
        goal = session.pln_ctx.enforce_bounds(
            np.deg2rad(np.asarray(HOME_QS_DEG, dtype=np.float32)))
    if not session.pln_ctx.is_state_valid(goal):
        print(f'{label}: the pose is in collision')
        return None
    return label, goal


def plan_tube(session, cap_poses, next_tube):
    """The selected tube's run from the loaded cap poses: a dict, or None
    (with the reason printed) when an orientation was not solved or a
    pose / path collides with the REAL cell. Nothing moves, nothing is
    solved; ``next_tube`` (or None for home) is where the run ends.

    The placing uses the set of the tube's own pick orientation; everything
    after it is the fixed set.
    """
    orientations, fixed = cap_poses
    name = cap.orientation_of(session.rotmat)
    if name not in orientations:
        print(f'tube {session.tube}: no cap poses for the {name} pick '
              'orientation -- re-run tool/precompute_poses_cap.py')
        return None
    entry = orientations[name]
    ctx = session.pln_ctx
    for label, qs in (('drop', entry['drop']),
                      ('push start', fixed['push_top']),
                      ('pop', fixed['pop']),
                      ('re-pick', fixed['repick_top']),
                      ('discard', fixed['discard'])):
        if not ctx.is_state_valid(qs):
            print(f'tube {session.tube}: the {label} pose collides with the '
                  'cell as it is now')
            return None
    for label, path in (('placing', entry['place']),
                        ('slide-out and rise after the release',
                         entry['release']),
                        ('push', fixed['push']),
                        ('lever press', fixed['press']),
                        ('re-pick descent', fixed['repick'])):
        if not pkm.local_sequence_valid(ctx, path):
            print(f'tube {session.tube}: the {label} collides with the cell '
                  'as it is now')
            return None
    after = solve_after(session, next_tube)
    if after is None:
        return None
    return {'orientation': name, 'drop_pos': cap.ORIENTATIONS[name][1],
            'drop': entry['drop'], 'place': entry['place'],
            'release': entry['release'], **fixed, 'after': after}


def print_plan(plan):
    deg = lambda qs: np.round(np.degrees(qs), 2)
    print(f'  {"drop":14s} {deg(plan["drop"])} deg  ({plan["orientation"]}, '
          f'grasp_center at {np.round(plan["drop_pos"] * 1000, 2)} mm)')
    print(f'  {"placed":14s} {deg(plan["place"][-1])} deg  '
          f'({cap.PLACE_DESCEND * 1000:.1f} mm down)')
    print(f'  {"push start":14s} {deg(plan["push_top"])} deg  '
          f'({np.round(cap.PUSH_POS * 1000, 1)} mm)')
    print(f'  {"pushed":14s} {deg(plan["push"][-1])} deg  '
          f'({cap.PUSH_DESCEND * 1000:.1f} mm down)')
    print(f'  {"pop":14s} {deg(plan["pop"])} deg  '
          f'(grasp_center at {np.round(cap.POP_POS * 1000, 2)} mm)')
    print(f'  {"pop pressed":14s} {deg(plan["press"][-1])} deg  '
          f'({cap.POP_DIST * 1000:.1f} mm down)')
    print(f'  {"re-pick":14s} {deg(plan["repick_top"])} deg  '
          f'(grasp_center at {np.round(cap.REPICK_POS * 1000, 2)} mm)')
    print(f'  {"discard":14s} {deg(plan["discard"])} deg  '
          f'(grasp_center at {np.round(cap.DISCARD_POS * 1000, 2)} mm)')
    after_label, after_qs = plan['after']
    print(f'  then -> {after_label}: {deg(after_qs)} deg')


def tube_steps(session, capper, plan):
    """The selected tube's full run as ``(label, step)`` pairs."""
    drop, place, push = plan['drop'], plan['place'], plan['push']
    press, after_label, after_qs = plan['press'], *plan['after']
    return session.pick_steps(free_speed=FREE_SPEED,
                              near_speed=NEAR_SPEED) + [
        # step 7
        ('move to the drop point',
         lambda: session.transfer_to('the drop point', drop,
                                     speed=CARRY_SPEED)),
        # step 8
        (f'lower {cap.PLACE_DESCEND * 1000:.0f} mm -- place the tube',
         lambda: session.follow_path('placed', place, speed=NEAR_SPEED)),
        # step 9
        ('open the gripper -- let go of the tube',
         lambda: session.work_gripper('open')),
        # step 10
        (f'slide {cap.RELEASE_SHIFT * 1000:.0f} mm along the TCP x, then '
         f'rise {cap.RELEASE_RISE * 1000:.0f} mm',
         lambda: session.follow_path('released', plan['release'],
                                     speed=NEAR_SPEED)),
        # step 11
        ('close the gripper', lambda: session.work_gripper('close')),
        # step 12
        ('move to the push start (push orientation)',
         lambda: session.transfer_to('the push start', plan['push_top'],
                                     speed=NEAR_SPEED)),
        # step 13
        (f'push {cap.PUSH_DESCEND * 1000:.0f} mm down at '
         f'{PUSH_SPEED:.0f}%',
         lambda: session.follow_path('pushed', push, speed=PUSH_SPEED)),
        # step 14
        ('retract to the push start',
         lambda: session.follow_path('the push start', push[::-1],
                                     speed=FREE_SPEED)),
        # step 15
        ('move to the pop position -- the capper opens, presses the cap '
         'and goes to the stopper meanwhile',
         lambda: pop_and_cap(session, capper, plan['pop'])),
        # step 16
        (f'press {cap.POP_DIST * 1000:.0f} mm down on the lever at '
         f'{POP_SPEED:.0f}%',
         lambda: session.follow_path('the lever pressed', press,
                                     speed=POP_SPEED)),
        # step 17
        ('retract from the lever',
         lambda: session.follow_path('the pop position', press[::-1],
                                     speed=NEAR_SPEED)),
        # step 18
        ('move the capper back home', lambda: capper_home(capper)),
        # step 19
        ('open the gripper', lambda: session.work_gripper('open')),
        # step 20
        ('move to the re-pick point',
         lambda: session.transfer_to('the re-pick point',
                                     plan['repick_top'], speed=NEAR_SPEED)),
        # step 21
        (f'lower {cap.REPICK_DESCEND * 1000:.0f} mm to the popped tube',
         lambda: session.follow_path('the popped tube', plan['repick'],
                                     speed=NEAR_SPEED)),
        # step 22
        ('close the gripper -- grip the popped tube',
         lambda: session.work_gripper('close')),
        # step 23
        ('retract to the re-pick point with the tube',
         lambda: session.follow_path('the re-pick point',
                                     plan['repick'][::-1],
                                     speed=NEAR_SPEED)),
        # step 24
        ('carry it to the discard point',
         lambda: session.transfer_to('the discard point', plan['discard'],
                                     speed=CARRY_SPEED)),
        # step 25
        ('open the gripper -- discard the tube',
         lambda: session.work_gripper('open')),
        # step 26
        (f'move to {after_label}',
         lambda: session.transfer_to(after_label, after_qs,
                                     speed=FREE_SPEED)),
    ]


def main(tube=TUBE_INDEX):
    try:
        session = PickSession(tube)
    except RuntimeError as exc:
        print(exc)
        return False
    cap_poses = load_cap_poses(session)
    next_tube = tube + 1 if tube + 1 < len(session.paths) else None
    plan = None if cap_poses is None else plan_tube(session, cap_poses,
                                                    next_tube)
    capper = None
    if plan is not None:
        try:
            capper = XH430CapperExec().connect()
        except DynamixelError as exc:
            print(f'capper: {exc}')
    if capper is None:
        session.release()
        return False
    print_plan(plan)
    try:
        return session.run(tube_steps(session, capper, plan))
    finally:
        capper.close()                  # the port only: a pressed cap stays


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tube', type=int, default=TUBE_INDEX,
                        help=f'precomputed pick index (default {TUBE_INDEX})')
    args = parser.parse_args()
    sys.exit(0 if main(tube=args.tube) else 1)
