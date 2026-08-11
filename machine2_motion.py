"""Machine 2 (the COBOTTA at 192.168.0.102) visits the tubes one after another.

Both arms are shown and both START FROM THEIR REAL JOINT ANGLES. Machine 2 is
the actor; machine 1 is read once and then stands still as an obstacle -- its
pose is baked into the collision model, so it is read BEFORE the model is
compiled and the connection is dropped afterwards.

The tube grid, the TCP orientation and the four-pose sequence all come from
``coripps.main``. Press ``space`` for the next tube: the motion is PREVIEWED as
an animation first and sent to the controller only once the animation has run
through. One tube is a transfer to ``pre_pick_high`` followed by the local
``pre_pick -> pick_place -> lift``. The gripper is not used yet -- the arm just
goes to the poses.

The transfer to the FIRST tube is planned with RRT-Connect, because the arm
starts wherever it was left. Tube to tube the straight joint-space line is
taken when it is collision-free, and RRT-Connect is the fallback.

Keys: ``space`` runs the next tube, ``q``/``esc`` quits.
"""

import os
import sys
import time

import numpy as np
import pyglet.window.key as pkey

# Allow running as `python coripps/machine2_motion.py` from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
import one.collider.mj_collider as ocm
import one.motion.core.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
from one.drivers.cobotta import BCapError, Cobotta, CobottaMotion

import coripps.main as cmain
from coripps.robots.mobisys import mobisys


# This PC is 192.168.0.100; machine 1 is the leaf sampler, machine 2 the tube
# handler.
_MACHINE1_HOST = '192.168.0.101'
_MACHINE2_HOST = '192.168.0.102'
_EXEC_SPEED = 100.0                      # external speed/accel/decel, percent
_CD_STEP_SIZE = np.pi / 180             # edge collision-check resolution
_EXTEND_STEP_SIZE = np.pi / 36          # RRT extension step
_MAX_ITERS = 3000
_SHORTCUT_ITERS = 200
_PREVIEW_STEP = np.pi / 60              # preview animation resolution
_PREVIEW_INTERVAL = 0.03
_START_TOLERANCE = np.deg2rad(2.0)      # real-vs-planned start mismatch guard
_TUBE_Z = 0.07                          # grasp_center height over the tube grid
_TUBE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'models', 'tube.stl')


def connect(host, label, motion):
    """Connect to one controller; return None (and say so) when unreachable."""
    cls = CobottaMotion if motion else Cobotta
    kwargs = {'speed': _EXEC_SPEED} if motion else {}
    try:
        cobotta = cls(host, **kwargs).connect()
    except (OSError, BCapError) as exc:
        print(f'{label}: cannot reach {host} ({exc})')
        return None
    print(f'{label}: connected to {host}')
    return cobotta


def build_scene():
    """Return the world plus the mobile system bodies."""
    base = ovw.World(cam_pos=(1.25, 0.75, 0.65),
                     cam_lookat_pos=(0.45, -0.10, 0.12))
    ossop.frame().attach_to(base.scene)
    mobisys_base, machine2, machine1 = mobisys()
    mobisys_base.attach_to(base.scene)
    machine2.attach_to(base.scene)
    machine2.gripper.attach_to(base.scene)
    machine2.gripper.toggle_tcp('grasp_center', color_mat=ouc.CoordColor.MYC)
    machine1.attach_to(base.scene)
    return base, mobisys_base, machine2, machine1


def build_planning_context(machine2, machine1):
    """Machine 2 is the actor; machine 1 and the ground are obstacles.

    Machine 1's pose is frozen into the MuJoCo model here, so it has to be at
    its real angles already. The tubes are deliberately NOT obstacles -- the
    arm has to reach into them.
    """
    ground = ossop.plane()
    collider = ocm.MJCollider()
    collider.append(machine2)
    collider.append(machine1)
    collider.append(ground)
    collider.actors = [machine2]
    collider.compile()
    return omppc.PlanningContext(collider=collider,
                                 cd_step_size=_CD_STEP_SIZE), ground


def solve_tubes(machine2):
    """Every reachable tube on the grid, with its four-pose joint sequence."""
    points = cmain.tube_picker_pick_points_xy(z=_TUBE_Z)
    valid_points, sequences, invalid_points = \
        cmain.filter_valid_point_sequences(machine2, points, cmain._TCP_ROTMAT)
    print(f'tubes: {len(valid_points)} reachable of {len(points)} grid points '
          f'({len(invalid_points)} without a usable IK sequence)')
    return valid_points, sequences


def show_tubes(scene, points):
    """One translucent tube model per reachable grid point."""
    template = osso.SceneObject.from_file(_TUBE_PATH, collision_type=None,
                                          rgb=ouc.ExtendedColor.SILVER,
                                          alpha=0.5)
    for point in points:
        tube = template.clone()
        tube.set_pos_rotmat(pos=point + cmain._TUBE_POS_OFFSET,
                            rotmat=cmain._TUBE_ROTMAT)
        tube.attach_to(scene)


def plan_transfer(pln_ctx, start, goal):
    """RRT-Connect plus shortcutting. None when it fails."""
    planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx,
                                     extend_step_size=_EXTEND_STEP_SIZE)
    t0 = time.time()
    path = planner.solve(start=start, goal=goal, max_iters=_MAX_ITERS)
    elapsed = time.time() - t0
    if not path:
        print(f'  transfer: no path after {elapsed:.3f}s')
        return None
    path = ompr.shortcut_path(path, pln_ctx, n_iter=_SHORTCUT_ITERS)
    print(f'  transfer: planned, {len(path)} waypoints in {elapsed:.3f}s')
    return path


def build_transfer(pln_ctx, start, goal, force_plan):
    """Get from ``start`` to ``goal``.

    The straight joint-space line when it is free -- tube to tube stays inside
    the pick region, so it usually is. RRT-Connect otherwise, and always for
    the first tube, whose start is wherever the arm happens to have been left.
    """
    if not force_plan and pln_ctx.is_motion_valid(start, goal):
        print('  transfer: direct, the straight line is free')
        return [start, goal]
    return plan_transfer(pln_ctx, start, goal)


def local_sequence_valid(pln_ctx, seq):
    """The approach/lift steps are short straight lines, but still checked."""
    for q0, q1 in zip(seq[:-1], seq[1:]):
        if not pln_ctx.is_motion_valid(q0, q1):
            return False
    return True


def build_program(transfer, seq):
    """The joint waypoints for one tube.

    ``transfer`` already ends at ``seq[0]`` (pre_pick_high), so what is left is
    the local descent ``pre_pick -> pick_place -> lift``.
    """
    return list(transfer) + [seq[1], seq[2], seq[3]]


def main():
    base, mobisys_base, machine2, machine1 = build_scene()

    # machine 1 is only ever read: one pose, then the socket goes away. It is
    # baked into the collision model, so a later move of it would NOT be seen.
    reader = connect(_MACHINE1_HOST, 'machine1', motion=False)
    if reader is not None:
        try:
            machine1.fk(qs=reader.joint_angles(ndof=machine1.ndof))
            print('machine1: frozen at its real pose as an obstacle -- '
                  'restart this script if it moves')
        except (OSError, BCapError) as exc:
            print(f'machine1: read failed ({exc}) -- left at its sim pose')
        reader.close()
    else:
        print('machine1: left at its sim pose as an obstacle')

    cobotta = connect(_MACHINE2_HOST, 'machine2', motion=True)

    pln_ctx, ground = build_planning_context(machine2, machine1)
    tube_points, tube_sequences = solve_tubes(machine2)
    if len(tube_points) == 0:
        print('tubes: nothing reachable -- nothing to do')
        if cobotta is not None:
            cobotta.close()
        return
    show_tubes(base.scene, tube_points)

    if cobotta is not None:
        machine2.fk(qs=cobotta.joint_angles(ndof=machine2.ndof))
    else:
        # the sim rest pose (all zeros) is itself in collision, so there would
        # be nothing to plan from -- park the arm at the first tube
        machine2.fk(qs=tube_sequences[0][0])
        print('machine2: no controller -- animation only, nothing is sent')

    cursor = ossop.sphere(pos=tube_points[0], radius=0.009, segments=12,
                          rgb=ouc.BasicColor.YELLOW, alpha=1.0)
    cursor.attach_to(base.scene)

    # 'start'/'program'/'preview' are swapped wholesale by select_tube(), so
    # they live in the state dict rather than in closure variables.
    state = {'mode': 'idle', 'i': 0, 'tube': -1, 'start': None,
             'program': None, 'preview': None}

    def current_pose():
        """Where the next transfer has to start from."""
        if cobotta is not None:
            return cobotta.joint_angles(ndof=machine2.ndof)
        if state['program'] is not None:
            # no controller: pretend the last program ran, otherwise the start
            # would be wherever the preview animation happens to have stopped
            return state['program'][-1]
        return np.asarray(machine2.qs, dtype=np.float32)

    def select_tube():
        """Plan the next tube, walking forward past the ones that fail.
        False when none of them works; the previous program is then kept."""
        try:
            start = pln_ctx.enforce_bounds(current_pose())
        except (OSError, BCapError) as exc:
            print(f'plan: lost {cobotta.host} ({exc})')
            return False
        if not pln_ctx.is_state_valid(start):
            print('plan: the current pose is in collision -- cannot plan '
                  'from it')
            return False
        first = state['tube'] < 0           # nothing run yet: always plan
        for offset in range(len(tube_points)):
            tube = (state['tube'] + 1 + offset) % len(tube_points)
            seq = tube_sequences[tube]
            print(f'tube {tube + 1}/{len(tube_points)}: '
                  f'point={np.round(tube_points[tube], 4)}')
            if not local_sequence_valid(pln_ctx, seq):
                print('  approach is in collision -- skipping this tube')
                continue
            transfer = build_transfer(pln_ctx, start, seq[0], force_plan=first)
            if transfer is None:
                print('  unreachable from here -- skipping this tube')
                continue
            program = build_program(transfer, seq)
            state['start'] = start
            state['tube'] = tube
            state['program'] = program
            state['preview'] = ompr.densify_path(program, pln_ctx,
                                                 max_step=_PREVIEW_STEP)
            state['mode'] = 'preview'
            state['i'] = 0
            cursor.set_pos_rotmat(pos=tube_points[tube])
            print(f"  {len(program)} waypoints, previewing "
                  f"{len(state['preview']) * _PREVIEW_INTERVAL:.1f}s")
            return True
        print('plan: no tube could be reached from here')
        return False

    def start_execution():
        """Called once the preview animation has run through."""
        if cobotta is None:
            machine2.fk(qs=state['program'][-1])
            state['mode'] = 'idle'
            print("  no controller -- 'space' for the next tube")
            return
        try:
            real_qs = cobotta.joint_angles(ndof=machine2.ndof)
        except (OSError, BCapError) as exc:
            print(f'execute: lost {cobotta.host} ({exc})')
            state['mode'] = 'idle'
            return
        drift = np.max(np.abs(real_qs - state['start']))
        if drift > _START_TOLERANCE:
            print(f'execute: the arm moved since planning '
                  f'({np.degrees(drift):.2f} deg off the planned start) -- '
                  f"refusing to run ('space' replans from here)")
            state['mode'] = 'idle'
            return
        try:
            cobotta.enable()                # a no-op once already powered
        except (OSError, BCapError) as exc:
            print(f'execute: cannot take the arm ({exc})')
            state['mode'] = 'idle'
            return
        state['mode'] = 'execute'
        state['i'] = 0
        print(f'  executing at {_EXEC_SPEED:.0f}%')

    def step_execution():
        program = state['program']
        i = state['i']
        if i >= len(program):
            # the arm stays taken and powered between tubes: a motor cycle
            # costs a brake release. cobotta.close() releases it on exit.
            state['mode'] = 'idle'
            print(f"  tube {state['tube'] + 1} reached, arm still live -- "
                  "'space' for the next one")
            return
        try:
            # one Move per tick; the last one stops precisely (@0), the rest
            # pass through (@P)
            cobotta.move_qs(program[i], pass_motion=(i < len(program) - 1))
            machine2.fk(qs=cobotta.joint_angles(ndof=machine2.ndof))
        except (OSError, BCapError) as exc:
            # release on failure, so the next enable() runs ClearError
            print(f'execute: move {i} failed ({exc}) -- stopping')
            cobotta.disable()
            state['mode'] = 'idle'
            return
        state['i'] = i + 1

    def step_preview():
        preview = state['preview']
        if state['i'] >= len(preview):
            start_execution()               # animation done, now the real arm
            return
        machine2.fk(qs=preview[state['i']])
        state['i'] += 1

    def tick(dt):
        # never re-trigger while the animation or the arm is running
        if state['mode'] == 'preview':
            step_preview()
            return
        if state['mode'] == 'execute':
            step_execution()
            return
        if base.input_manager.is_key_pressed_edge(pkey.SPACE):
            select_tube()

    base.schedule_interval(tick, interval=_PREVIEW_INTERVAL)
    print("ready: 'space' animates the move to the next tube, then runs it")
    return base, cobotta, machine1, machine2, tube_points, tube_sequences, state


if __name__ == '__main__':
    import builtins

    result = main()
    if result is not None:
        (base, cobotta, machine1, machine2,
         tube_points, tube_sequences, state) = result
        builtins.base = base
        builtins.cobotta = cobotta
        builtins.machine1 = machine1
        builtins.machine2 = machine2
        builtins.tube_points = tube_points
        builtins.tube_sequences = tube_sequences
        builtins.state = state
        try:
            base.run()
        finally:
            if cobotta is not None:
                cobotta.close()
