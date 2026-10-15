"""Look at the plant with the cutter hand's D405, find the leaves, pick one,
and line the cutter's tcp up with it.

1. observe: the sampler moves the D405 to OBSERVE_HEIGHT straight above
   VIEW_CENTER (sampler base frame, where the plant goes), looking down --
   planned, printed, 'g' in the terminal moves it. One frame (color +
   depth) is taken and saved to ``observe/`` with the joint angles.
2. leaves (``leaves``): SAM 3 finds the plant and its leaves; a frame per
   leaf, the leaf with the most room at its petiole.
3. over the plant: the tcp onto the chosen leaf's frame moved to the plant
   centre R (R: the rosette centre's camera ray at the height of the leaf's
   C) -- first PRE_APPROACH_OFFSET back along z (IK nearest the observation
   pose, the straight joint line when free, RRT-Connect otherwise), then
   straight down to RELOOK_OFFSET; at EXEC_SPEED. The viewer loops the move
   with the point cloud, C / J and the frames; 'g' in the viewer window
   runs it (the real joints are checked against the planned start first).
4. look again from there (closer): a new frame, the leaves found again, the
   chosen one = the leaf whose C is nearest the old C; its frame replaces
   the old one (how much it moved is printed). Saved to ``observe/`` as
   ``relook_*``.
5. onto the leaf: straight from over R to RELOOK_OFFSET over the new C,
   then straight down along the leaf's z to APPROACH_OFFSET; at
   DESCENT_SPEED. The viewer switches to this move; 'g' runs it.

The collision model is the sampler, the picker (read once, frozen) and the
ground -- NOT the plant or its stand.

    py -3.12 coripps/vision/d405/leaf_pipeline.py
    py -3.12 coripps/vision/d405/leaf_pipeline.py --offline   # the saved frame
"""

import os
import sys
import json
import time
import argparse
import threading

import cv2
import numpy as np

import one.motion.probabilistic.rrt as ompr
import one.utils.math as oum
from one.drivers.cobotta import BCapError, Cobotta, CobottaMotion

from coripps.tool import pose_sequence as pseq
from coripps.tool.home_sampler import _build, _SAMPLER_HOST, _PICKER_HOST
from coripps.tool.keys import _wait_for_g
from coripps.vision.d405 import leaves as lv
from coripps.vision.d405.camera import D405
from coripps.vision.d405.calib import HANDEYE_PATH


_HERE = os.path.dirname(os.path.abspath(__file__))
OBSERVE_DIR = os.path.join(_HERE, 'observe')
VIEW_CENTER = (0.220, -0.060, 0.050)    # sampler base frame (m)
OBSERVE_HEIGHT = 0.200                  # camera above VIEW_CENTER (m)
ROLL_STEP_DEG = 15
PRE_APPROACH_OFFSET = 0.050             # m, back along the leaf's z
RELOOK_OFFSET = 0.020                   # m, ditto, where it looks again
APPROACH_OFFSET = 0.010                 # m, ditto, where the descent ends
DESCENT_STEP = 0.002                    # m between descent samples
TURN_STEP = np.deg2rad(2.0)             # rad, most turn between samples
SPEED = 30.0                            # %, moving to observe
EXEC_SPEED = 20.0                       # %, the transfer
DESCENT_SPEED = 10.0                    # %, the straight descent
ARRIVAL_TOLERANCE = np.deg2rad(0.5)
SETTLE = 0.5                            # s before the frame
PLAY_STEP = np.deg2rad(1.0)             # rad per drawn frame
PLAY_INTERVAL = 1 / 30.0
HOLD_FRAMES = 30
POLL_INTERVAL = 0.1                     # s, following the real sampler
FRAME_LENGTH = 0.02                     # m, drawn leaf frames
ROI_PX = 40                             # drawn cloud: this far round leaves
_EXTEND_STEP_SIZE = np.pi / 36
_MAX_ITERS = 3000
_SHORTCUT_ITERS = 200


def load_handeye():
    with open(HANDEYE_PATH, encoding='utf-8') as f:
        return np.array(json.load(f)['tf_flange_cam'])


def tf_w_cam(sampler, qs, tf_flange_cam):
    sampler.fk(qs=np.asarray(qs, dtype=np.float32))
    return (np.array(sampler.tcp('flange').tf, dtype=np.float64)
            @ tf_flange_cam)


def nearest_valid(pln_ctx, sols, near):
    valid = [np.asarray(q, np.float32) for q in sols
             if pln_ctx.is_state_valid(np.asarray(q, np.float32))]
    if not valid:
        return None
    return min(valid, key=lambda q: np.max(np.abs(q - near)))


def plan(pln_ctx, start, goal):
    """A collision-free joint path, or None."""
    if pln_ctx.is_motion_valid(start, goal):
        return [start, goal]
    planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx,
                                     extend_step_size=_EXTEND_STEP_SIZE)
    path = planner.solve(start=start, goal=goal, max_iters=_MAX_ITERS)
    if not path:
        return None
    return ompr.shortcut_path(path, pln_ctx, n_iter=_SHORTCUT_ITERS)


# -- 1. observe ----------------------------------------------------------
def observe_goal(sampler, pln_ctx, tf_flange_cam, near):
    """Joints putting the D405 OBSERVE_HEIGHT straight above VIEW_CENTER,
    looking down; of the rolls the collision-free IK nearest ``near``."""
    sampler.fk(qs=near)
    tf_w_base = np.array(sampler.tf, dtype=np.float64)
    center = (tf_w_base @ np.r_[VIEW_CENTER, 1.0])[:3]
    pos = center + np.array([0.0, 0.0, OBSERVE_HEIGHT])
    tf_cam_flange = np.linalg.inv(tf_flange_cam)
    best = None
    for roll in range(0, 360, ROLL_STEP_DEG):
        a = np.deg2rad(roll)
        x = np.array([np.cos(a), np.sin(a), 0.0])
        z = np.array([0.0, 0.0, -1.0])
        tf_w_flange = lv.tf(pos, np.column_stack([x, np.cross(z, x), z])) \
            @ tf_cam_flange
        qs = nearest_valid(pln_ctx, sampler.ik(
            tf_w_flange[:3, 3].astype(np.float32),
            tf_w_flange[:3, :3].astype(np.float32), ref_qs=near), near)
        if qs is not None and (best is None or np.max(np.abs(qs - near))
                               < np.max(np.abs(best - near))):
            best = qs
    return best


def observe(sampler, pln_ctx, tf_flange_cam):
    """Move to the observation pose, take a frame, save it. ``(color,
    depth, K, qs)`` or None."""
    try:
        cobotta = CobottaMotion(_SAMPLER_HOST, speed=SPEED).connect()
    except (OSError, BCapError) as exc:
        print(f'sampler: cannot reach {_SAMPLER_HOST} ({exc})')
        return None
    try:
        start = pln_ctx.enforce_bounds(
            cobotta.joint_angles(ndof=sampler.ndof))
        if not pln_ctx.is_state_valid(start):
            print('start: in collision in the model -- refusing to move')
            return None
        goal = observe_goal(sampler, pln_ctx, tf_flange_cam, start)
        if goal is None:
            print('observe: no collision-free IK above VIEW_CENTER')
            return None
        path = plan(pln_ctx, start, goal)
        if path is None:
            print('observe: no collision-free path')
            return None
        print(f'observe: D405 {OBSERVE_HEIGHT * 1000:.0f} mm above '
              f'{np.round(np.array(VIEW_CENTER) * 1000)} mm (sampler base), '
              f'joints {np.round(np.degrees(goal), 1)} deg, '
              f'{len(path)} waypoints at {SPEED:.0f}%')
        if not _wait_for_g('observe'):
            return None
        cobotta.enable()
        cobotta.move_path(path)
        time.sleep(SETTLE)
        qs = cobotta.joint_angles(ndof=sampler.ndof)
    except (OSError, BCapError) as exc:
        print(f'sampler: {exc} -- stopping')
        return None
    finally:
        cobotta.close()                 # motors off, arm given back
    cam = D405()
    try:
        color, depth, _ = cam.frames()
        K = cam.K
    finally:
        cam.close()
    save_frame(color, depth, K, qs)
    return color, depth, K, qs


def save_frame(color, depth, K, qs, prefix=''):
    """A frame and where it was taken from, to ``observe/``:
    ``<prefix>color.png``, ``<prefix>depth.npy``, ``<prefix>observe.json``
    (joints, K) -- read back by :func:`load_observation`."""
    os.makedirs(OBSERVE_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OBSERVE_DIR, f'{prefix}color.png'), color)
    np.save(os.path.join(OBSERVE_DIR, f'{prefix}depth.npy'), depth)
    with open(os.path.join(OBSERVE_DIR, f'{prefix}observe.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'qs': [float(q) for q in qs], 'K': np.asarray(K).tolist()},
                  f, indent=2)


def load_observation(prefix=''):
    """``(color, depth, K, qs)`` saved by :func:`save_frame`."""
    with open(os.path.join(OBSERVE_DIR, f'{prefix}observe.json'),
              encoding='utf-8') as f:
        rec = json.load(f)
    return (cv2.imread(os.path.join(OBSERVE_DIR, f'{prefix}color.png')),
            np.load(os.path.join(OBSERVE_DIR, f'{prefix}depth.npy')),
            np.array(rec['K']), np.array(rec['qs'], np.float32))


# -- 3.-5. approach -----------------------------------------------------
def back_along_z(leaf_tf, offset):
    """The leaf frame moved ``offset`` back along its own z."""
    tgt = np.array(leaf_tf, dtype=np.float64)
    tgt[:3, 3] -= offset * tgt[:3, 2]
    return tgt


def plant_frame(leaf, rosette_px, K, cam_tf):
    """The leaf's frame moved to the plant centre: R's camera ray at the
    height of the leaf's C."""
    o, v = lv.ray(rosette_px, cam_tf, K)
    r = o + (leaf['C'][2] - o[2]) / v[2] * v
    frame = np.array(leaf['tf'], dtype=np.float64)
    frame[:3, 3] = r
    return frame


def to_plant(sampler, pln_ctx, plant_tf, start):
    """Step 3's path, or None (said why)."""
    pre = back_along_z(plant_tf, PRE_APPROACH_OFFSET)
    goal = nearest_valid(pln_ctx, sampler.ik(
        pre[:3, 3].astype(np.float32), pre[:3, :3].astype(np.float32),
        tcp=sampler.cutter.tcp('tcp'), ref_qs=start), start)
    if goal is None:
        print(f'no collision-free IK {PRE_APPROACH_OFFSET * 1000:.0f} mm over '
              'the plant centre')
        return None
    path = plan(pln_ctx, start, goal)
    if path is None:
        print('no collision-free path to over the plant centre')
        return None
    n = int(round((PRE_APPROACH_OFFSET - RELOOK_OFFSET) / DESCENT_STEP))
    down = straight(sampler, pln_ctx, goal,
                    [back_along_z(plant_tf, PRE_APPROACH_OFFSET - (
                        PRE_APPROACH_OFFSET - RELOOK_OFFSET) * k / n)[:3, 3]
                     for k in range(1, n + 1)],
                    plant_tf[:3, :3], 'down over the plant centre')
    return None if down is None else list(path) + down[1:]


def straight(sampler, pln_ctx, qs, positions, rotmat, what):
    """A straight tcp path through ``positions`` (``qs`` first), or None
    (said why). ``rotmat``: one orientation for all, or one per position
    (the tcp turning on the way)."""
    rotmats = (rotmat if np.ndim(rotmat) == 3
               else [rotmat] * len(positions))
    path = [np.asarray(qs, np.float32)]
    for pos, rot in zip(positions, rotmats):
        step = pseq.follow_positions(sampler, path[-1],
                                     [np.asarray(pos, np.float32)],
                                     np.asarray(rot, np.float32),
                                     tcp=sampler.cutter.tcp('tcp'))
        if step is None:
            print(f'{what}: no smooth IK')
            return None
        path.append(step[-1])
    if not all(pln_ctx.is_motion_valid(a, b)
               for a, b in zip(path[:-1], path[1:])):
        print(f'{what}: in collision')
        return None
    return list(path)


def to_leaf(sampler, pln_ctx, leaf_tf, qs):
    """Step 5's path (one list), or None (said why)."""
    sampler.fk(qs=qs)
    cur_tf = np.array(sampler.cutter.tcp('tcp').tf, dtype=np.float64)
    cur = cur_tf[:3, 3]
    pre = back_along_z(leaf_tf, RELOOK_OFFSET)[:3, 3]
    turn = np.arccos(np.clip((np.trace(cur_tf[:3, :3].T @ leaf_tf[:3, :3])
                              - 1) / 2, -1, 1))
    # the position and the orientation both on the way, evenly: as many
    # samples as the longer of the two needs
    n = max(int(np.ceil(np.linalg.norm(pre - cur) / DESCENT_STEP)),
            int(np.ceil(turn / TURN_STEP)), 1)
    rots = oum.rotmat_slerp(cur_tf[:3, :3], leaf_tf[:3, :3], n + 1)[1:]
    over = straight(sampler, pln_ctx, qs,
                    [cur + (pre - cur) * k / n for k in range(1, n + 1)],
                    rots, 'over the leaf')
    if over is None:
        return None
    n = int(round((RELOOK_OFFSET - APPROACH_OFFSET) / DESCENT_STEP))
    down = straight(sampler, pln_ctx, over[-1],
                    [back_along_z(leaf_tf, RELOOK_OFFSET - (
                        RELOOK_OFFSET - APPROACH_OFFSET) * k / n)[:3, 3]
                     for k in range(1, n + 1)],
                    leaf_tf[:3, :3], 'down to the leaf')
    return None if down is None else over + down[1:]


def densify(path):
    frames = [np.asarray(path[0], np.float32)]
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        n = max(int(np.ceil(np.max(np.abs(b - a)) / PLAY_STEP)), 1)
        frames += [a + (b - a) * k / n for k in range(1, n + 1)]
    return frames


def run_path(path, speed, what):
    """Run ``path`` on the real sampler; the arm given back after. False
    when it did not (said why)."""
    try:
        cobotta = CobottaMotion(_SAMPLER_HOST, speed=speed).connect()
    except (OSError, BCapError) as exc:
        print(f'sampler: cannot reach {_SAMPLER_HOST} ({exc})')
        return False
    try:
        off = np.max(np.abs(cobotta.joint_angles(ndof=len(path[0]))
                            - path[0]))
        if off > ARRIVAL_TOLERANCE:
            print(f'sampler: moved since planning ({np.degrees(off):.2f} '
                  'deg off the planned start) -- not running')
            return False
        cobotta.enable()
        print(f'sampler: {what}, {len(path)} waypoints at {speed:.0f}%')
        cobotta.move_path(path)
        drift = np.max(np.abs(cobotta.joint_angles(ndof=len(path[0]))
                              - path[-1]))
        print(f'sampler: arrived, {np.degrees(drift):.3f} deg off the goal')
        return True
    except (OSError, BCapError) as exc:
        print(f'sampler: {exc} -- stopped')
        return False
    finally:
        cobotta.close()


def relook(job, old):
    """Step 4: the chosen leaf seen again from where the arm is; the new
    leaf dict, or None (said why)."""
    time.sleep(SETTLE)
    qs = read_qs(_SAMPLER_HOST, 'sampler', job['sampler'].ndof)
    if qs is None:
        return None, None
    cam = D405()
    try:
        color, depth, _ = cam.frames()
        K = cam.K
    finally:
        cam.close()
    cam_tf = tf_w_cam(job['sampler'], qs, job['tf_flange_cam'])
    save_frame(color, depth, K, qs, prefix='relook_')
    found = lv.leaf_frames(color, depth, K, cam_tf, job['segmenter'])
    if found is None or not found[0]:
        print('relook: no leaves found')
        return None, qs
    leaves, rosette = found
    k = int(np.argmin([np.linalg.norm(leaf['C'] - old['C'])
                       for leaf in leaves]))
    new = leaves[k]
    cv2.imwrite(os.path.join(OBSERVE_DIR, 'relook_leaves.png'),
                lv.draw(color, leaves, rosette, k))
    turn = np.degrees(np.arccos(np.clip(
        (np.trace(old['tf'][:3, :3].T @ new['tf'][:3, :3]) - 1) / 2, -1, 1)))
    print(f"relook: the leaf is leaf {k} now; C moved "
          f"{np.round((new['C'] - old['C']) * 1000, 2)} mm, J moved "
          f"{np.round((new['J'] - old['J']) * 1000, 2)} mm, frame turned "
          f"{turn:.2f} deg")
    return new, qs


def stage_b(job, status):
    """In a thread: step 3 on the arm, then steps 4 and 5's planning."""
    if not run_path(job['path_a'], EXEC_SPEED, 'over the plant centre'):
        status['state'] = 'failed'
        return
    new, qs = relook(job, job['leaf'])
    if new is None:
        status['state'] = 'failed'
        return
    path = to_leaf(job['sampler'], job['pln_ctx'], new['tf'],
                   job['pln_ctx'].enforce_bounds(qs))
    if path is None:
        status['state'] = 'failed'
        return
    print(f'onto the leaf: {len(path) - 1} steps to '
          f"{np.round(back_along_z(new['tf'], APPROACH_OFFSET)[:3, 3] * 1000, 1)}"
          f" mm ({APPROACH_OFFSET * 1000:.0f} mm over the leaf)")
    job['path_b'] = path
    job['frames'] = densify(path)
    status['state'] = 'preview_b'
    print("viewer: 'g' in the window runs the move onto the leaf")


def stage_c(job, status):
    """In a thread: step 5 on the arm."""
    ok = run_path(job['path_b'], DESCENT_SPEED, 'onto the leaf')
    status['state'] = 'done' if ok else 'failed'


def read_qs(host, label, ndof=6):
    try:
        with Cobotta(host) as cobotta:
            return cobotta.joint_angles(ndof=ndof)
    except (OSError, BCapError) as exc:
        print(f'{label}: cannot read {host} ({exc})')
        return None


def cloud(color, depth, K, tf_w_cam, leaves):
    """Colored world points around the leaves (for the viewer)."""
    us = np.concatenate([[leaf['C_px'][0], leaf['J_px'][0]] for leaf in leaves])
    vs = np.concatenate([[leaf['C_px'][1], leaf['J_px'][1]] for leaf in leaves])
    h, w = depth.shape
    roi = np.zeros(depth.shape, bool)
    roi[max(int(vs.min()) - ROI_PX, 0):min(int(vs.max()) + ROI_PX, h),
        max(int(us.min()) - ROI_PX, 0):min(int(us.max()) + ROI_PX, w)] = True
    pts = lv.depth_points(roi, depth, K, tf_w_cam)
    v, u = np.nonzero(roi & (depth > 0))
    return pts, color[v, u, ::-1] / 255.0


def show(job, leaves, chosen, plant_tf, pts_rgb):
    """The viewer: preview step 3, 'g' runs it and steps 4 and 5's planning,
    preview step 5, 'g' runs it."""
    from pyglet.window import key
    import one.scene.scene_object_primitive as ossop
    import one.utils.constant as ouc
    import one.viewer.world as ovw
    from coripps.robots.mobisys import mobisys

    center = leaves[chosen]['C']
    base = ovw.World(cam_pos=center + np.array([0.55, 0.45, 0.40]),
                     cam_lookat_pos=center + np.array([0.0, 0.0, 0.10]))
    mobisys_base, picker, sampler = mobisys()
    picker_qs = read_qs(_PICKER_HOST, 'picker')
    if picker_qs is not None:
        picker.fk(qs=picker_qs)
    sampler.fk(qs=job['frames'][0])
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    sampler.cutter.toggle_tcp('tcp', length_scale=FRAME_LENGTH / 0.2 * 2,
                              radius_scale=FRAME_LENGTH / 0.2 * 2)
    ossop.point_cloud(*pts_rgb).attach_to(base.scene)
    scale = FRAME_LENGTH / 0.2          # the frame primitive is 0.2 m long
    for k, leaf in enumerate(leaves):
        ossop.sphere(pos=leaf['C'], radius=0.001,
                     rgb=ouc.BasicColor.RED).attach_to(base.scene)
        ossop.sphere(pos=leaf['J'], radius=0.001,
                     rgb=ouc.BasicColor.BLUE).attach_to(base.scene)
        ossop.frame_from_tf(leaf['tf'], length_scale=scale * (2 if k == chosen
                                                              else 1),
                            radius_scale=scale * 2).attach_to(base.scene)
    ossop.frame_from_tf(plant_tf, length_scale=scale * 2,
                        radius_scale=scale * 2).attach_to(base.scene)
    status = {'state': 'preview_a'}
    state = {'i': 0, 'frames': None}
    reader = {}
    print("viewer: 'g' in the window runs the move over the plant centre")

    def follow():
        if 'cobotta' not in reader:
            try:
                reader['cobotta'] = Cobotta(_SAMPLER_HOST).connect()
            except (OSError, BCapError):
                return
        try:
            sampler.fk(qs=reader['cobotta'].joint_angles(ndof=sampler.ndof))
        except (OSError, BCapError):
            reader.pop('cobotta').close()

    def tick(dt):
        st = status['state']
        if st in ('preview_a', 'preview_b'):
            if state['frames'] is not job['frames']:     # a new preview
                state['frames'], state['i'] = job['frames'], 0
                state['timeline'] = (
                    [0] * HOLD_FRAMES + list(range(len(job['frames'])))
                    + [len(job['frames']) - 1] * HOLD_FRAMES)
            if base.input_manager.is_key_pressed_edge(key.G):
                status['state'] = ('running_a' if st == 'preview_a'
                                   else 'running_b')
                target = stage_b if st == 'preview_a' else stage_c
                threading.Thread(target=target, args=(job, status),
                                 daemon=True).start()
                return
            sampler.fk(qs=state['frames'][state['timeline'][state['i']]])
            state['i'] = (state['i'] + 1) % len(state['timeline'])
        else:
            follow()                    # running, done or failed: the arm

    base.schedule_interval(tick, interval=PLAY_INTERVAL)
    try:
        base.run()
    finally:
        if 'cobotta' in reader:
            reader['cobotta'].close()


def main(offline=False):
    tf_flange_cam = load_handeye()
    print('loading SAM 3 ...')
    segmenter = lv.Segmenter()
    _, _, sampler, pln_ctx = _build()   # reads and freezes the picker
    print('== 1. observe')
    if offline:
        color, depth, K, qs = load_observation()
        print(f'the saved frame (joints {np.round(np.degrees(qs), 1)} deg)')
    else:
        got = observe(sampler, pln_ctx, tf_flange_cam)
        if got is None:
            return False
        color, depth, K, qs = got
    cam = tf_w_cam(sampler, qs, tf_flange_cam)

    print('\n== 2. leaves')
    found = lv.leaf_frames(color, depth, K, cam, segmenter)
    if found is None or not found[0]:
        print('no leaves found')
        return False
    leaves, rosette = found
    for k, leaf in enumerate(leaves):
        slope = np.degrees(np.arcsin(np.clip(leaf['tf'][2, 1], -1, 1)))
        print(f"leaf {k}: C {np.round(leaf['C'] * 1000, 1)} mm  C-J "
              f"{np.linalg.norm(leaf['C'] - leaf['J']) * 1000:.1f} mm  J->C "
              f"rises {slope:+.1f} deg  room at J "
              f"{leaf['room'][0]:.1f} / {leaf['room'][1]:.1f} mm")
    chosen = lv.choose(leaves)
    print(f'chosen: leaf {chosen}')
    os.makedirs(OBSERVE_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OBSERVE_DIR, 'leaves.png'),
                lv.draw(color, leaves, rosette, chosen))

    print('\n== 3. over the plant centre')
    start = read_qs(_SAMPLER_HOST, 'sampler', sampler.ndof)
    if start is None:
        return False
    start = pln_ctx.enforce_bounds(start)
    plant_tf = plant_frame(leaves[chosen], rosette, K, cam)
    path = to_plant(sampler, pln_ctx, plant_tf, start)
    if path is None:
        return False
    print(f'plant centre R {np.round(plant_tf[:3, 3] * 1000, 1)} mm; '
          f'{len(path)} waypoints to '
          f'{np.round(back_along_z(plant_tf, RELOOK_OFFSET)[:3, 3] * 1000, 1)}'
          f' mm ({PRE_APPROACH_OFFSET * 1000:.0f} mm over it, then straight '
          f'down to {RELOOK_OFFSET * 1000:.0f} mm)')
    job = {'sampler': sampler, 'pln_ctx': pln_ctx, 'segmenter': segmenter,
           'tf_flange_cam': tf_flange_cam, 'leaf': leaves[chosen],
           'path_a': path, 'frames': densify(path)}
    show(job, leaves, chosen, plant_tf, cloud(color, depth, K, cam, leaves))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--offline', action='store_true',
                        help='use the frame saved in observe/, do not move '
                             'to observe')
    args = parser.parse_args()
    sys.exit(0 if main(offline=args.offline) else 1)
