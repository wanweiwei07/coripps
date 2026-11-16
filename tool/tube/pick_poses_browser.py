"""Look at the tubes ``precompute_poses_pick`` kept, one at a time.

Reads ``config/poses_pick.json`` -- it does NOT re-solve, so it opens in a
second instead of a minute, and what you see is exactly the data the
execution scripts use.

The whole grid is drawn: a tube at every candidate, a green dot where the
pick solved and a red one where it did not. ``space`` steps to the next
kept tube, driving the arm to its stored ``pre_pick`` (the hand looking at
the cap from above) and parking a yellow cursor on that TCP position. The
printed ``tube k`` is the index ``main.py --tube k`` / ``single_routine.py
--tube k`` take. (The grip itself depends on the cap's long tab, so it is
solved at run time and not in the file.)

    py -3.12 coripps/tool/tube/pick_poses_browser.py
"""

import os

import numpy as np
import pyglet.window.key as pkey

import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

import coripps.tool.tube.layout_pick as layout
import coripps.tool.tube.poses_io_pick as ppio
from coripps.robots.mobisys import mobisys


_TUBE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'models', 'tube.stl')


def browse(path=ppio.PICK_POSES_JSON):
    points, look_qs, meta = ppio.load_checked(path, layout=layout)
    print(f"{len(points)} tubes from {os.path.basename(path)} "
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
        np.any(np.all(np.isclose(points, point), axis=1))
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

    cursor = ossop.sphere(pos=(0.0, 0.0, 0.0), radius=0.009, segments=12,
                          rgb=ouc.BasicColor.YELLOW, alpha=1.0)
    cursor.attach_to(base.scene)

    state = {'tube': 0}

    def show_current():
        k = state['tube']
        tube_handler.fk(qs=look_qs[k])
        tcp = layout.pre_pick_pos(points[k])
        cursor.set_pos_rotmat(pos=tcp)
        base.scene.dirty = True
        print(f'tube {k} ({k + 1}/{len(points)}) pre_pick '
              f'tcp={np.round(tcp, 4)}')

    def step(dt):
        if not base.input_manager.is_key_pressed_edge(pkey.SPACE):
            return
        state['tube'] = (state['tube'] + 1) % len(points)
        show_current()

    show_current()
    base.schedule_interval(step, interval=0.03)
    print(f'grid {len(grid)}: {int(solved.sum())} green, '
          f'{int((~solved).sum())} red -- space steps to the next tube')
    return base, tube_handler, points, look_qs, state


if __name__ == '__main__':
    import builtins

    base, tube_handler, points, look_qs, state = browse()
    builtins.base = base
    builtins.tube_handler = tube_handler
    builtins.points = points
    builtins.look_qs = look_qs
    builtins.state = state
    base.run()
