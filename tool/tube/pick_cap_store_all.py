"""Run ``pick_cap_store`` on every tube, first to last, one step per key.

Headless: no viewer. The arm, the gripper and the capper are connected once
and stay connected across tubes. Each tube is the full ``pick_cap_store``
run (pick, place, push, cap, pop, discard); its last step moves straight on
to the next tube's pre_pick_high, and after the last tube to home.

Before anything moves, EVERY tube is planned from ``config/poses_cap.json``
(see ``precompute_poses_cap``): a tube that cannot be run (its orientation
was not solved, or a pose collides with the cell as it is now) is skipped
and listed, and the tube before it moves on to the next one that can.
Nothing has moved yet, so skipping is harmless.

Each step waits for ``g``; with ``--c`` they run back to back without
waiting. Either way ``q`` (or Esc / Ctrl+C) stops -- in ``--c`` mode before
the next step, once the one in progress has finished. ANY failure
stops the whole run and releases the hardware -- the arm may be standing in
the store or holding a tube, so it is left for you to sort out by hand. The
printout says which tube to restart from with ``--start``.

    py -3.12 coripps/tool/tube/pick_cap_store_all.py
    py -3.12 coripps/tool/tube/pick_cap_store_all.py --start 12
    py -3.12 coripps/tool/tube/pick_cap_store_all.py --c          # no 'g'
"""

import sys
import argparse

from coripps.tool.tube.pick_sequence_single import PickSession
from coripps.end_effectors.xh430_capper.exec import (XH430CapperExec,
                                                     DynamixelError)
import coripps.tool.tube.pick_cap_store as pcs


def plan_all(session, cap_poses, start):
    """Plan every tube from ``start`` on: ``[(tube, plan), ...]`` for the
    runnable ones, each ending at the NEXT runnable tube (home after the
    last)."""
    runnable = []
    for tube in range(start, len(session.paths)):
        try:
            session.select(tube, verbose=False)
        except RuntimeError as exc:
            print(f'  skip tube {tube}: {exc}')
            continue
        if pcs.plan_tube(session, cap_poses, None) is None:
            print(f'  skip tube {tube}: no plan (see above)')
            continue
        runnable.append(tube)
    plans = []
    for k, tube in enumerate(runnable):
        session.select(tube, verbose=False)
        next_tube = runnable[k + 1] if k + 1 < len(runnable) else None
        plan = pcs.plan_tube(session, cap_poses, next_tube)
        if plan is None:                # its hand-over pose is the problem
            print(f'  skip tube {tube}: cannot move on to '
                  f'{"home" if next_tube is None else f"tube {next_tube}"}')
            continue
        plans.append((tube, plan))
    return plans


def main(start=0, continuous=False):
    try:
        session = PickSession(start)
    except RuntimeError as exc:
        print(exc)
        return False
    cap_poses = pcs.load_cap_poses(session)
    if cap_poses is None:
        session.release()
        return False
    print(f'planning tubes {start}..{len(session.paths) - 1} ...')
    plans = plan_all(session, cap_poses, start)
    if not plans:
        print('no tube can be run')
        session.release()
        return False
    print(f'{len(plans)} tube(s) to run: {[tube for tube, _ in plans]}')
    if continuous:
        print("--c: running without 'g' -- press q to stop before the next "
              'step')
    try:
        capper = XH430CapperExec().connect()
    except DynamixelError as exc:
        print(f'capper: {exc}')
        session.release()
        return False

    ok = True
    try:
        for k, (tube, plan) in enumerate(plans):
            session.select(tube)
            pcs.print_plan(plan)
            title = f'tube {tube} ({k + 1}/{len(plans)})'
            ok = session.run(pcs.tube_steps(session, capper, plan),
                             release=False, title=title,
                             continuous=continuous)
            if not ok:
                print(f'\ntube {tube} FAILED -- everything stopped. Sort the '
                      'cell out by hand, then restart with '
                      f'--start {tube}')
                break
            if session.quit:
                print(f'\nstopped during tube {tube}. Restart with '
                      f'--start {tube} (or the next tube if it finished)')
                break
            print(f'\n=== tube {tube} done ===')
        else:
            print('\nall tubes done')
    finally:
        session.release()
        capper.close()                  # the port only: a pressed cap stays
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--start', type=int, default=0,
                        help='first precomputed pick index to run (default 0)')
    parser.add_argument('--c', action='store_true',
                        help="continuous: run every step without waiting "
                             "for 'g' (q still stops)")
    args = parser.parse_args()
    sys.exit(0 if main(start=args.start, continuous=args.c) else 1)
