"""Send the picker (the COBOTTA at 192.168.0.102) to its home joint pose.

Home: the gripper's grasp_center at (100, -100, 100) mm in the picker's base
frame, pointing straight down (the start / retreat pose of
``coripps.single_routine``), as joint angles.

Headless: no viewer, nothing to press. The sampler is read once and baked into
the collision model as a standing obstacle, exactly as in
``coripps.picker_motion``. The move is the straight joint-space line when it
is free, RRT-Connect otherwise, and nothing is sent when neither works.

    py -3.12 coripps/tool/tube/home_picker.py
"""

import sys
import time

import numpy as np

import one.scene.scene_object_primitive as ossop
import one.collider.mj_collider as ocm
import one.motion.core.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
from one.drivers.cobotta import BCapError, Cobotta, CobottaMotion

from coripps.robots.mobisys import mobisys


_SAMPLER_HOST = '192.168.0.101'
_PICKER_HOST = '192.168.0.102'
_EXEC_SPEED = 30.0                      # homing transit, so slower than a pick
_CD_STEP_SIZE = np.pi / 180
_EXTEND_STEP_SIZE = np.pi / 36
_MAX_ITERS = 3000
_SHORTCUT_ITERS = 200
_ARRIVAL_TOLERANCE = np.deg2rad(0.5)
# grasp_center at (100, -100, 100) mm (base frame), pointing down
HOME_QS_DEG = (-28.47, -4.24, 133.43, 0.0, 50.81, -58.47)


def _build():
    _, picker, sampler = mobisys()

    # the sampler only ever gets read: one pose, then the socket goes
    # away. It is frozen into the collision model, so a later move of it
    # would NOT be seen.
    try:
        reader = Cobotta(_SAMPLER_HOST).connect()
    except (OSError, BCapError) as exc:
        print(f'sampler: cannot reach {_SAMPLER_HOST} ({exc}) -- '
              'left at its sim pose as an obstacle')
    else:
        try:
            qs1 = reader.joint_angles(ndof=sampler.ndof)
            sampler.fk(qs=qs1)
            print(f'sampler: frozen as an obstacle at '
                  f'{np.round(np.degrees(qs1), 2)} deg')
        except (OSError, BCapError) as exc:
            print(f'sampler: read failed ({exc}) -- left at its sim pose')
        reader.close()

    ground = ossop.plane()
    collider = ocm.MJCollider()
    collider.append(picker)
    collider.append(sampler)
    collider.append(ground)
    collider.actors = [picker]
    collider.compile()
    pln_ctx = omppc.PlanningContext(collider=collider,
                                    cd_step_size=_CD_STEP_SIZE)
    return ground, collider, picker, pln_ctx


def home():
    """Plan and run the picker home. True when it arrived."""
    _, _, picker, pln_ctx = _build()

    wanted = np.deg2rad(np.asarray(HOME_QS_DEG, dtype=np.float32))
    goal = pln_ctx.enforce_bounds(wanted)
    if np.any(np.abs(goal - wanted) > _ARRIVAL_TOLERANCE):
        print(f'goal: {np.round(HOME_QS_DEG, 2)} deg is out of the joint '
              f'limits -- going to {np.round(np.degrees(goal), 2)} deg instead')
    if not pln_ctx.is_state_valid(goal):
        print('goal: the home pose is in collision -- refusing to move')
        return False

    try:
        cobotta = CobottaMotion(_PICKER_HOST, speed=_EXEC_SPEED).connect()
    except (OSError, BCapError) as exc:
        print(f'picker: cannot reach {_PICKER_HOST} ({exc})')
        return False
    print(f'picker: connected to {_PICKER_HOST}')

    try:
        start = pln_ctx.enforce_bounds(
            cobotta.joint_angles(ndof=picker.ndof))
        print(f'picker: at {np.round(np.degrees(start), 2)} deg')
        if not pln_ctx.is_state_valid(start):
            print('start: the current pose is in collision -- cannot plan '
                  'from it')
            return False
        if np.max(np.abs(start - goal)) <= _ARRIVAL_TOLERANCE:
            print('picker: already home -- nothing to do')
            return True

        if pln_ctx.is_motion_valid(start, goal):
            print('path: direct, the straight line is free')
            path = [start, goal]
        else:
            planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx,
                                             extend_step_size=_EXTEND_STEP_SIZE)
            t0 = time.time()
            path = planner.solve(start=start, goal=goal, max_iters=_MAX_ITERS)
            elapsed = time.time() - t0
            if not path:
                print(f'path: no route home after {elapsed:.3f}s -- '
                      'nothing sent')
                return False
            path = ompr.shortcut_path(path, pln_ctx, n_iter=_SHORTCUT_ITERS)
            print(f'path: planned, {len(path)} waypoints in {elapsed:.3f}s')

        cobotta.enable()
        print(f'picker: running at {_EXEC_SPEED:.0f}%')
        cobotta.move_path(
            path,
            on_waypoint=lambda i, qs: print(
                f'  {i + 1}/{len(path)} {np.round(np.degrees(qs), 2)}'))
        reached = cobotta.joint_angles(ndof=picker.ndof)
        drift = np.max(np.abs(reached - goal))
        print(f'picker: arrived at {np.round(np.degrees(reached), 3)} deg '
              f'({np.degrees(drift):.3f} deg off home)')
        return drift <= _ARRIVAL_TOLERANCE
    except (OSError, BCapError) as exc:
        print(f'picker: {exc} -- stopping')
        cobotta.disable()
        return False
    finally:
        # close() powers the motors off and gives the arm back
        cobotta.close()


if __name__ == '__main__':
    sys.exit(0 if home() else 1)
