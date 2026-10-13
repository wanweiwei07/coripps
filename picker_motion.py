"""The picker (the COBOTTA at 192.168.0.102) visits the tubes in turn.

Both arms are shown and both START FROM THEIR REAL JOINT ANGLES. The picker is
the actor; the sampler is read once and then stands still as an obstacle -- its
pose is baked into the collision model, so it is read BEFORE the model is
compiled and the connection is dropped afterwards.

The tube grid, the TCP orientation and the four-pose sequence all come from
``coripps.layout_pick`` / ``pose_sequence``. Press ``space`` for the next
tube: the
motion is PREVIEWED as an animation first and sent to the controller only once
the animation has run through. One tube is::

    transfer -> pre_pick_high -> [OPEN gripper] -> pre_pick
             -> pick -> [CLOSE gripper] -> lift

The gripper opens above the tube, clear of it, and closes once the arm is
standing at ``pick``; the arm never passes through a gripper step. Without
the gripper servo the gripper steps are only simulated.

The transfer to the FIRST tube is planned with RRT-Connect, because the arm
starts wherever it was left. Tube to tube the straight joint-space line is
taken when it is collision-free, and RRT-Connect is the fallback.

Keys: ``space`` runs the next tube, ``q``/``esc`` quits.
"""

import os
import time

import numpy as np
import pyglet.window.key as pkey

import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
import one.collider.mj_collider as ocm
import one.motion.core.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
from one.drivers.cobotta import BCapError, Cobotta, CobottaMotion

import coripps.tool.layout_pick as layout
import coripps.tool.pose_sequence as pseq
import coripps.tool.poses_io_pick as ppio
from coripps.robots.mobisys import mobisys
from coripps.end_effectors.xh430_gripper import OPEN_TRAVEL
from coripps.end_effectors.xh430_gripper.exec import (XH430GripperExec,
                                                      DynamixelError)


# This PC is 192.168.0.100; the sampler is the leaf-sampling arm, the
# picker the tube handler.
_SAMPLER_HOST = '192.168.0.101'
_PICKER_HOST = '192.168.0.102'
_EXEC_SPEED = 100.0                      # external speed/accel/decel, percent
_CD_STEP_SIZE = np.pi / 180             # edge collision-check resolution
_EXTEND_STEP_SIZE = np.pi / 36          # RRT extension step
_MAX_ITERS = 3000
_SHORTCUT_ITERS = 200
_PREVIEW_STEP = np.pi / 60              # preview animation resolution
_PREVIEW_INTERVAL = 0.03
_START_TOLERANCE = np.deg2rad(2.0)      # real-vs-planned start mismatch guard
# the grid height comes from the layout, so a live solve and the
# precomputed file can never disagree about where the tubes are
_TUBE_Z = layout.PICK_Z
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
    mobisys_base, picker, sampler = mobisys()
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    picker.gripper.toggle_tcp('grasp_center', color_mat=ouc.CoordColor.MYC)
    sampler.attach_to(base.scene)
    return base, mobisys_base, picker, sampler


def freeze_sampler(sampler):
    """Read the sampler once and leave it standing there as an obstacle.

    It is only ever READ: one pose, then the socket goes away. The pose is
    baked into the collision model afterwards, so a later move of it would NOT
    be seen -- restart the script if it moves. Shared with the single-tube
    pick script.
    """
    reader = connect(_SAMPLER_HOST, 'sampler', motion=False)
    if reader is None:
        print('sampler: left at its sim pose as an obstacle')
        return
    try:
        sampler.fk(qs=reader.joint_angles(ndof=sampler.ndof))
        print('sampler: frozen at its real pose as an obstacle -- '
              'restart this script if it moves')
    except (OSError, BCapError) as exc:
        print(f'sampler: read failed ({exc}) -- left at its sim pose')
    reader.close()


def build_planning_context(picker, sampler):
    """The picker is the actor; the sampler and the ground are obstacles.

    The sampler's pose is frozen into the MuJoCo model here, so it has to be at
    its real angles already. The tubes are deliberately NOT obstacles -- the
    arm has to reach into them.
    """
    ground = ossop.plane()
    collider = ocm.MJCollider()
    collider.append(picker)
    collider.append(sampler)
    collider.append(ground)
    collider.actors = [picker]
    collider.compile()
    return omppc.PlanningContext(collider=collider,
                                 cd_step_size=_CD_STEP_SIZE), ground


def solve_tubes(picker):
    """Every reachable tube on the grid, with its straight-line joint path."""
    points = layout.pick_points_xy(z=_TUBE_Z)
    valid_points, paths, marks, _rotmats, invalid_points = pseq.filter_valid(
        picker, points, layout.tcp_rotmats, layout.pose_path)
    print(f'tubes: {len(valid_points)} reachable of {len(points)} grid points '
          f'({len(invalid_points)} without a usable IK sequence)')
    return valid_points, paths, marks


def show_tubes(scene, points):
    """One translucent tube model per reachable grid point."""
    template = osso.SceneObject.from_file(_TUBE_PATH, collision_type=None,
                                          rgb=ouc.ExtendedColor.SILVER,
                                          alpha=0.5)
    for point in points:
        tube = template.clone()
        tube.set_pos_rotmat(pos=point + ppio.TUBE_POS_OFFSET,
                            rotmat=ppio.TUBE_ROTMAT)
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
    """Every sample-to-sample step of the approach / lift path is checked."""
    for q0, q1 in zip(seq[:-1], seq[1:]):
        if not pln_ctx.is_motion_valid(q0, q1):
            return False
    return True


def build_program(transfer, path, marks):
    """One tube as a typed step list: ``('arm', qs)`` or ``('gripper', verb)``.

    ``transfer`` already ends at ``path[marks[0]]`` (pre_pick_high). The
    gripper opens there, before the descent, and closes at ``pick``, before
    the lift. Between key poses the arm plays every straight-line sample of
    ``path``, so the TCP moves in straight lines, not joint-space arcs.
    """
    m_high, _m_pre, m_pick, m_lift = (int(m) for m in marks)
    steps = [('arm', q) for q in transfer]     # ... ends at pre_pick_high
    steps.append(('gripper', 'open'))
    steps += [('arm', q) for q in path[m_high + 1:m_pick + 1]]   # -> pick
    steps.append(('gripper', 'close'))
    steps += [('arm', q) for q in path[m_pick + 1:m_lift + 1]]   # -> lift
    return steps


def densify_with_marks(path, pln_ctx, max_step=_PREVIEW_STEP):
    """Densify ``path`` for the animation; ``marks[i]`` is the dense index of
    original waypoint ``i``, so the preview can work the gripper at exactly
    the poses the real run will."""
    dense = [np.asarray(path[0], dtype=np.float32)]
    marks = [0]
    for start, goal in zip(path[:-1], path[1:]):
        segment = ompr.densify_path([start, goal], pln_ctx, max_step=max_step)
        dense.extend(segment[1:])
        marks.append(len(dense) - 1)
    return dense, marks


def main(tube_source=solve_tubes):
    """Run the interactive tube tour.

    ``tube_source(picker)`` supplies ``(points, paths, marks)``; the default
    solves the grid here, ``picker_execute_picks`` passes a loader for the
    precomputed ``config/poses_pick.json`` instead.
    """
    base, mobisys_base, picker, sampler = build_scene()
    freeze_sampler(sampler)
    cobotta = connect(_PICKER_HOST, 'picker', motion=True)
    try:
        gripper = XH430GripperExec().connect()
    except DynamixelError as exc:
        gripper = None
        print(f'{exc}\n         the gripper steps will be simulated only')

    pln_ctx, ground = build_planning_context(picker, sampler)
    tube_points, tube_paths, marks = tube_source(picker)
    if len(tube_points) == 0:
        print('tubes: nothing reachable -- nothing to do')
        if cobotta is not None:
            cobotta.close()
        if gripper is not None:
            gripper.close()
        return
    show_tubes(base.scene, tube_points)

    if cobotta is not None:
        picker.fk(qs=cobotta.joint_angles(ndof=picker.ndof))
    else:
        # the sim rest pose (all zeros) is itself in collision, so there would
        # be nothing to plan from -- park the arm at the first tube
        picker.fk(qs=tube_paths[0][marks[0]])
        print('picker: no controller -- animation only, nothing is sent')

    cursor = ossop.sphere(pos=tube_points[0], radius=0.009, segments=12,
                          rgb=ouc.BasicColor.YELLOW, alpha=1.0)
    cursor.attach_to(base.scene)

    # the sim gripper is shown at the angle the REAL one is configured for, so
    # preview and execution agree even if the working open is retuned
    open_deg = (gripper.cfg.open_deg if gripper is not None
                else float(np.degrees(OPEN_TRAVEL)))
    open_width = picker.gripper.opening_from_angle(np.deg2rad(open_deg))
    shut_width = float(picker.gripper.jaw_range[0])

    # 'start'/'program'/'preview'/the frames are swapped wholesale by
    # select_tube(), so they live in the state dict rather than in closures.
    state = {'mode': 'idle', 'i': 0, 'tube': -1, 'start': None,
             'program': None, 'preview': None,
             'open_frame': 0, 'grip_frame': 0}

    def show_gripper(width):
        picker.gripper.set_opening(width)
        base.scene.dirty = True

    def sync_gripper_from_real():
        """Mirror the servo's own angle onto the sim gripper."""
        width = np.clip(gripper.opening_mm() / 1000.0,
                        picker.gripper.jaw_range[0],
                        picker.gripper.jaw_range[1])
        show_gripper(float(width))

    def current_pose():
        """Where the next transfer has to start from."""
        if cobotta is not None:
            return cobotta.joint_angles(ndof=picker.ndof)
        if state['program'] is not None:
            # no controller: pretend the last program ran, otherwise the start
            # would be wherever the preview animation happens to have stopped
            return next(q for kind, q in reversed(state['program'])
                        if kind == 'arm')
        return np.asarray(picker.qs, dtype=np.float32)

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
            seq = tube_paths[tube]
            print(f'tube {tube + 1}/{len(tube_points)}: '
                  f'point={np.round(tube_points[tube], 4)}')
            if not local_sequence_valid(pln_ctx, seq):
                print('  approach is in collision -- skipping this tube')
                continue
            transfer = build_transfer(pln_ctx, start, seq[marks[0]],
                                      force_plan=first)
            if transfer is None:
                print('  unreachable from here -- skipping this tube')
                continue
            program = build_program(transfer, seq, marks)
            # the preview animates the ARM waypoints; the marks say which
            # dense frame each landed on, so the sim gripper acts at the
            # same poses as the real one
            arm_path = [q for kind, q in program if kind == 'arm']
            dense, frame_marks = densify_with_marks(arm_path, pln_ctx)
            state['start'] = start
            state['tube'] = tube
            state['program'] = program
            state['preview'] = dense
            # arm_path = transfer + path after pre_pick_high, so pick sits
            # (m_pick - m_high) samples after the transfer's last waypoint
            at_high = len(transfer) - 1
            at_pick = at_high + int(marks[2]) - int(marks[0])
            state['open_frame'] = frame_marks[at_high]   # pre_pick_high
            state['grip_frame'] = frame_marks[at_pick]   # pick
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
            picker.fk(qs=current_pose())
            show_gripper(shut_width)
            state['mode'] = 'idle'
            print("  no controller -- 'space' for the next tube")
            return
        try:
            real_qs = cobotta.joint_angles(ndof=picker.ndof)
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
        kind, payload = program[i]
        if kind == 'gripper':
            if gripper is None:
                print(f'  step {i + 1}: gripper {payload} (no servo -- '
                      'simulated)')
                show_gripper(open_width if payload == 'open' else shut_width)
            else:
                try:
                    if payload == 'open':
                        gripper.open()
                    else:
                        gripper.close_jaws()
                    sync_gripper_from_real()
                except DynamixelError as exc:
                    print(f'execute: gripper {payload} failed ({exc}) -- '
                          'stopping')
                    state['mode'] = 'idle'
                    return
                print(f'  step {i + 1}: gripper {payload} -> '
                      f'{gripper.opening_mm():5.2f} mm @ '
                      f'{gripper.dxl.present_current_ma():4.0f} mA')
            state['i'] = i + 1
            return
        try:
            # one Move per tick; the last one stops precisely (@0), the rest
            # pass through (@P) -- but never into a gripper step: the arm has
            # to be standing still while the blades move
            nxt = program[i + 1][0] if i + 1 < len(program) else None
            cobotta.move_qs(payload, pass_motion=(nxt == 'arm'))
            picker.fk(qs=cobotta.joint_angles(ndof=picker.ndof))
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
        i = state['i']
        picker.fk(qs=preview[i])
        opened = state['open_frame'] <= i < state['grip_frame']
        show_gripper(open_width if opened else shut_width)
        state['i'] = i + 1

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
    return (base, cobotta, gripper, sampler, picker, tube_points,
            tube_paths, state)


def run(result):
    """Publish what ``main`` returned to ``builtins`` and enter the viewer
    loop, always releasing the controller on the way out. Shared with
    ``picker_execute_picks``."""
    import builtins

    if result is not None:
        (base, cobotta, gripper, sampler, picker,
         tube_points, tube_paths, state) = result
        builtins.base = base
        builtins.cobotta = cobotta
        builtins.gripper = gripper
        builtins.sampler = sampler
        builtins.picker = picker
        builtins.tube_points = tube_points
        builtins.tube_paths = tube_paths
        builtins.state = state
        try:
            base.run()
        finally:
            if cobotta is not None:
                cobotta.close()
            if gripper is not None:
                gripper.close()         # the port only: a held tube stays held


if __name__ == '__main__':
    run(main())
