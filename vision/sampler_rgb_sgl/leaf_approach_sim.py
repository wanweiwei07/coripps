"""Bring the sampler's cutter tcp onto the chosen leaf's frame
(``leaf_frames``), APPROACH_OFFSET back along the leaf's z: plan it, loop the
move in the viewer, and run it on the real sampler when 'g' is pressed in
the viewer window.

The target is the leaf frame moved by -APPROACH_OFFSET along its own z (z
runs from the leaf's face to its back, so this is just over the face), with
the cutter tcp's axes on the leaf frame's axes. IK for the sampler with the
cutter's 'tcp'; of the collision-free solutions the one nearest the real
current joints. The path from the real current joints: the straight
joint-space line when free, RRT-Connect otherwise -- against the same
collision model as ``home_sampler`` (sampler, the picker frozen at its real
pose, the ground).

The viewer loops the move; it shows the plant's point cloud, the chosen
leaf's frame and the cutter's tcp frame. 'g' (once): the real joints are
checked against the planned start (ARRIVAL_TOLERANCE), the path is run at
EXEC_SPEED, and the viewer then follows the real sampler instead of the
loop. NOTE: the plant and its stand are not in the collision model.

    py -3.12 coripps/vision/sampler_rgb_sgl/leaf_approach_sim.py
"""

import sys
import time
import threading

import numpy as np

import one.motion.probabilistic.rrt as ompr
from one.drivers.cobotta import BCapError, Cobotta, CobottaMotion

from coripps.tool.home_sampler import _build, _SAMPLER_HOST, _PICKER_HOST
from coripps.vision.sampler_rgb_sgl import calib
from coripps.vision.sampler_rgb_sgl import leaf_frames as lf
from coripps.vision.sampler_rgb_sgl import reconstruct_survey as rs


APPROACH_OFFSET = 0.010         # m, back along the leaf's z
PLAY_STEP = np.deg2rad(1.0)     # rad, largest joint step per drawn frame
PLAY_INTERVAL = 1 / 30.0        # s
HOLD_FRAMES = 30                # frames held at each end of the loop
EXEC_SPEED = 20.0               # %, on the real sampler
ARRIVAL_TOLERANCE = np.deg2rad(0.5)
POLL_INTERVAL = 0.1             # s, reading the real sampler while it runs
_EXTEND_STEP_SIZE = np.pi / 36
_MAX_ITERS = 3000
_SHORTCUT_ITERS = 200


def target_tf(leaf_tf):
    """The tcp target: the leaf frame moved -APPROACH_OFFSET along its z."""
    tf = np.array(leaf_tf, dtype=np.float64)
    tf[:3, 3] -= APPROACH_OFFSET * tf[:3, 2]
    return tf


def solve_goal(sampler, pln_ctx, tf, near):
    """Collision-free joints putting the cutter tcp on ``tf``, nearest
    ``near``; None when there are none."""
    sols = sampler.ik(tf[:3, 3].astype(np.float32),
                      tf[:3, :3].astype(np.float32),
                      tcp=sampler.cutter.tcp('tcp'), ref_qs=near)
    valid = [np.asarray(q, np.float32) for q in sols
             if pln_ctx.is_state_valid(np.asarray(q, np.float32))]
    if not valid:
        print(f'IK: {len(sols)} solutions, none collision-free and in limits'
              if sols else 'IK: the target is out of reach')
        return None
    return min(valid, key=lambda q: np.max(np.abs(q - near)))


def plan(pln_ctx, start, goal):
    """A collision-free joint path, or None."""
    if pln_ctx.is_motion_valid(start, goal):
        print('path: direct, the straight joint line is free')
        return [start, goal]
    planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx,
                                     extend_step_size=_EXTEND_STEP_SIZE)
    t0 = time.time()
    path = planner.solve(start=start, goal=goal, max_iters=_MAX_ITERS)
    if not path:
        print('path: none found')
        return None
    path = ompr.shortcut_path(path, pln_ctx, n_iter=_SHORTCUT_ITERS)
    print(f'path: RRT-Connect, {len(path)} waypoints in '
          f'{time.time() - t0:.2f}s')
    return path


def densify(path):
    """The path resampled so no joint moves more than PLAY_STEP between
    frames."""
    frames = [np.asarray(path[0], np.float32)]
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        n = max(int(np.ceil(np.max(np.abs(b - a)) / PLAY_STEP)), 1)
        frames += [a + (b - a) * k / n for k in range(1, n + 1)]
    return frames


def read_qs(host, label, ndof=6):
    try:
        with Cobotta(host) as cobotta:
            qs = cobotta.joint_angles(ndof=ndof)
    except (OSError, BCapError) as exc:
        print(f'{label}: cannot read {host} ({exc})')
        return None
    print(f'{label}: {np.round(np.degrees(qs), 2)} deg')
    return qs


def execute(path, status):
    """Run ``path`` on the real sampler (in a thread); ``status['state']``
    ends 'done' or 'failed'."""
    try:
        cobotta = CobottaMotion(_SAMPLER_HOST, speed=EXEC_SPEED).connect()
    except (OSError, BCapError) as exc:
        print(f'sampler: cannot reach {_SAMPLER_HOST} ({exc})')
        status['state'] = 'failed'
        return
    try:
        now = cobotta.joint_angles(ndof=len(path[0]))
        off = np.max(np.abs(now - path[0]))
        if off > ARRIVAL_TOLERANCE:
            print(f'sampler: moved since planning ({np.degrees(off):.2f} deg '
                  'off the planned start) -- not running; rerun the script')
            status['state'] = 'failed'
            return
        cobotta.enable()
        print(f'sampler: running {len(path)} waypoints at {EXEC_SPEED:.0f}%')
        cobotta.move_path(path)
        reached = cobotta.joint_angles(ndof=len(path[0]))
        drift = np.max(np.abs(reached - path[-1]))
        print(f'sampler: arrived, {np.degrees(drift):.3f} deg off the goal')
        status['state'] = 'done'
    except (OSError, BCapError) as exc:
        print(f'sampler: {exc} -- stopped')
        status['state'] = 'failed'
    finally:
        cobotta.close()                 # motors off, arm given back


def show(frames, path, leaf_tf, cloud, picker_qs):
    from pyglet.window import key
    import one.scene.scene_object_primitive as ossop
    import one.utils.constant as ouc
    import one.viewer.world as ovw
    from coripps.robots.mobisys import mobisys

    center = leaf_tf[:3, 3]
    base = ovw.World(cam_pos=center + np.array([0.55, 0.45, 0.40]),
                     cam_lookat_pos=center + np.array([0.0, 0.0, 0.10]))
    mobisys_base, picker, sampler = mobisys()
    if picker_qs is not None:
        picker.fk(qs=picker_qs)
    sampler.fk(qs=frames[0])
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    sampler.cutter.toggle_tcp('tcp', length_scale=lf.FRAME_LENGTH / 0.2 * 2,
                              radius_scale=lf.FRAME_LENGTH / 0.2 * 2)
    ossop.point_cloud(cloud[0], cloud[1]).attach_to(base.scene)
    ossop.frame_from_tf(leaf_tf, length_scale=lf.FRAME_LENGTH / 0.2,
                        radius_scale=lf.FRAME_LENGTH / 0.2 * 2,
                        color_mat=ouc.CoordColor.RGB).attach_to(base.scene)
    # play, hold at the end, hold at the start, again
    timeline = ([0] * HOLD_FRAMES + list(range(len(frames)))
                + [len(frames) - 1] * HOLD_FRAMES)
    state = {'i': 0}
    status = {'state': 'preview'}       # -> running -> done / failed
    print("viewer: 'g' in the window runs the move on the real sampler")

    def tick(dt):
        if status['state'] != 'preview':
            return
        if base.input_manager.is_key_pressed_edge(key.G):
            status['state'] = 'running'
            threading.Thread(target=execute, args=(path, status),
                             daemon=True).start()
            base.schedule_interval(follow, interval=POLL_INTERVAL)
            return
        sampler.fk(qs=frames[timeline[state['i']]])
        state['i'] = (state['i'] + 1) % len(timeline)

    reader = {}

    def follow(dt):
        """Show the real sampler while (and after) it runs."""
        if 'cobotta' not in reader:
            try:
                reader['cobotta'] = Cobotta(_SAMPLER_HOST).connect()
            except (OSError, BCapError):
                return
        try:
            sampler.fk(qs=reader['cobotta'].joint_angles(ndof=sampler.ndof))
        except (OSError, BCapError):
            reader.pop('cobotta').close()

    base.schedule_interval(tick, interval=PLAY_INTERVAL)
    try:
        base.run()
    finally:
        if 'cobotta' in reader:
            reader['cobotta'].close()


def main():
    images, cams = rs.load()
    K, D = calib.load_intrinsics()
    found = lf.leaf_frames(images, cams, K, D)
    if found is None or not found[0]:
        print('no leaves found')
        return False
    leaves, _, cloud = found
    chosen = lf.choose(leaves)
    leaf = leaves[chosen]
    print(f"leaf {chosen}: C {np.round(leaf['C'] * 1000, 1)} mm, room at J "
          f"{leaf['room'][0]:.1f} / {leaf['room'][1]:.1f} mm")
    tf = target_tf(leaf['tf'])
    print(f'tcp target: {np.round(tf[:3, 3] * 1000, 1)} mm '
          f'({APPROACH_OFFSET * 1000:.0f} mm off the leaf face)')

    _, _, sampler, pln_ctx = _build()   # reads and freezes the picker
    start = read_qs(_SAMPLER_HOST, 'sampler', sampler.ndof)
    if start is None:
        return False
    start = pln_ctx.enforce_bounds(start)
    if not pln_ctx.is_state_valid(start):
        print('start: the current pose is in collision in the model')
        return False
    goal = solve_goal(sampler, pln_ctx, tf, start)
    if goal is None:
        return False
    print(f'goal: {np.round(np.degrees(goal), 2)} deg')
    sampler.fk(qs=goal)
    reached = np.array(sampler.cutter.tcp('tcp').tf, dtype=np.float64)
    print(f'  tcp at the goal: {np.linalg.norm(reached[:3, 3] - tf[:3, 3]) * 1000:.3f} '
          f'mm, {np.degrees(np.arccos(np.clip((np.trace(reached[:3, :3].T @ tf[:3, :3]) - 1) / 2, -1, 1))):.3f} '
          'deg off the target')
    path = plan(pln_ctx, start, goal)
    if path is None:
        return False
    frames = densify(path)
    print(f'animation: {len(frames)} frames')
    show(frames, path, leaf['tf'], cloud, read_qs(_PICKER_HOST, 'picker'))
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
