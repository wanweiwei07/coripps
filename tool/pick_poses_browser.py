"""Look at what ``precompute_pick_poses`` produced, one pose at a time.

Reads ``config/pick_poses.json`` -- it does NOT re-solve, so it opens in a
second instead of a minute, and what you see is exactly the data the execution
scripts will replay, not a fresh IK solution that may have picked a different
branch.

The whole grid is drawn: a tube at every candidate, a green dot where the pick
solved and a red one where it did not. ``space`` steps through the four poses
of the selected pick and then on to the next pick, driving the arm to the
stored joint angles and parking a yellow cursor on the TCP position.

    py -3.12 coripps/tool/pick_poses_browser.py
"""

import os
import sys

import numpy as np
import pyglet.window.key as pkey

# Allow running as `py -3.12 coripps/tool/pick_poses_browser.py` from the root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

import coripps.pick_layout as layout
import coripps.pick_poses_io as ppio
from coripps.robots.mobisys import mobisys


_TUBE_PATH = os.path.join(_PROJECT_ROOT, 'coripps', 'models', 'tube.stl')


def browse(path=ppio.PICK_POSES_JSON):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing -- run tool/precompute_pick_poses.py first')
    valid_points, sequences, rotmats, meta = ppio.load(path)
    print(f"{len(valid_points)} picks from {os.path.basename(path)} "
          f"(grid {meta.get('grid_total')}, "
          f"grasp height {meta.get('pick_z')})")

    base = ovw.World(cam_pos=(1.25, 0.75, 0.65),
                     cam_lookat_pos=(0.45, -0.10, 0.12))
    ossop.frame().attach_to(base.scene)
    mobisys_base, tube_handler, leaf_sampler = mobisys()
    mobisys_base.attach_to(base.scene)
    tube_handler.attach_to(base.scene)
    tube_handler.gripper.attach_to(base.scene)
    tube_handler.gripper.toggle_tcp('grasp_center', color_mat=ouc.CoordColor.MYC)
    leaf_sampler.attach_to(base.scene)

    # the file holds only what SOLVED, so the failures are the grid minus it
    grid = layout.pick_points_xy(z=meta.get('pick_z', layout.PICK_Z))
    solved = np.asarray([
        np.any(np.all(np.isclose(valid_points, point), axis=1))
        for point in grid], dtype=bool)

    tube_template = osso.SceneObject.from_file(
        _TUBE_PATH, collision_type=None, rgb=ouc.ExtendedColor.SILVER,
        alpha=0.5)
    for point in grid:
        tube = tube_template.clone()
        tube.set_pos_rotmat(pos=point + ppio.TUBE_POS_OFFSET,
                            rotmat=ppio.TUBE_ROTMAT)
        tube.attach_to(base.scene)
    for point, ok in zip(grid, solved):
        ossop.sphere(pos=point, radius=0.0035, segments=8, alpha=0.95,
                     rgb=ouc.BasicColor.GREEN if ok else ouc.BasicColor.RED
                     ).attach_to(base.scene)

    cursor = ossop.sphere(
        pos=valid_points[0] if len(valid_points) else (0.0, 0.0, 0.0),
        radius=0.009, segments=12, rgb=ouc.BasicColor.YELLOW, alpha=1.0)
    cursor.attach_to(base.scene)

    state = {'point_idx': 0, 'pose_idx': 0}

    def show_current():
        if len(valid_points) == 0:
            return
        point_idx, pose_idx = state['point_idx'], state['pose_idx']
        # the orientation this point actually solved with, not a re-guess
        rotmat = rotmats[point_idx]
        points = layout.pose_points(valid_points[point_idx], rotmat)
        tube_handler.fk(qs=sequences[point_idx, pose_idx])
        cursor.set_pos_rotmat(pos=points[pose_idx])
        base.scene.dirty = True
        print(f'pick {point_idx + 1}/{len(valid_points)} '
              f'pose {pose_idx + 1}/{len(ppio.POSE_NAMES)} '
              f'{ppio.POSE_NAMES[pose_idx]} '
              f'{"no_yaw" if layout.is_no_yaw(rotmat) else "yaw"} '
              f'tcp={np.round(points[pose_idx], 4)}')

    def step(dt):
        if not base.input_manager.is_key_pressed_edge(pkey.SPACE):
            return
        if len(valid_points) == 0:
            return
        state['pose_idx'] += 1
        if state['pose_idx'] >= len(ppio.POSE_NAMES):
            state['pose_idx'] = 0
            state['point_idx'] = (state['point_idx'] + 1) % len(valid_points)
        show_current()

    show_current()
    base.schedule_interval(step, interval=0.03)
    print(f'grid {len(grid)}: {int(solved.sum())} green, '
          f'{int((~solved).sum())} red -- space steps through the poses')
    return base, tube_handler, valid_points, sequences, rotmats, state


if __name__ == '__main__':
    import builtins

    (base, tube_handler, valid_points, sequences,
     rotmats, state) = browse()
    builtins.base = base
    builtins.tube_handler = tube_handler
    builtins.valid_points = valid_points
    builtins.sequences = sequences
    builtins.rotmats = rotmats
    builtins.state = state
    base.run()
