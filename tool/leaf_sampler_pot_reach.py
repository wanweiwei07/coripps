"""Can the leaf sampler look into every pot?

For each configured pot it solves the D405 camera pose 170 mm above the pot
with the camera axis pointing straight down, and draws the answer: a green dot
plus a translucent arm at the solution, a red dot where there is none.

This is about the SAMPLER and the pots; it has nothing to do with the tube picks
and deliberately imports none of that -- the two studies are independent and
either can be thrown away without touching the other.

    py -3.12 coripps/tool/leaf_sampler_pot_reach.py
"""

import os

import numpy as np

import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from coripps.robots.mobisys import mobisys


_POT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'models', 'pot.stl')
_POT_ROTMAT = np.eye(3, dtype=np.float32)

_MM = 1e-3
_POT_XS_MM = (380, 480, 580, 680, 780)
_POT_Y_MM = -120
_POT_Z_MM = -43
_POT_OBSERVE_Z_OFFSET_MM = 170          # camera stand-off above the pot
_LOOK_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _mm_pos(x, y, z):
    return np.array([x, y, z], dtype=np.float32) * _MM


def pot_positions():
    """The configured pot positions, in meters."""
    return np.asarray([_mm_pos(x, _POT_Y_MM, _POT_Z_MM) for x in _POT_XS_MM],
                      dtype=np.float32)


def pot_observation_points():
    """Where the camera has to sit to look into each pot."""
    return np.asarray(
        [_mm_pos(x, _POT_Y_MM, _POT_Z_MM + _POT_OBSERVE_Z_OFFSET_MM)
         for x in _POT_XS_MM], dtype=np.float32)


def show_observations(scene, leaf_sampler, points):
    """Draw reachable / unreachable D405 observation points."""
    results = []
    for idx, point in enumerate(points, start=1):
        sols, infos = leaf_sampler.ik_partial(
            tgt_pos=point,
            axis_constraints={'z': _LOOK_DOWN},
            chain='to_j5',
            tcp='d405',
            seed_count=128,
            return_infos=True,
        )
        ok = len(sols) > 0
        qs = sols[0] if ok else None
        info = infos[0] if infos else None
        ossop.sphere(pos=point, radius=0.007, segments=12, alpha=1.0,
                     rgb=ouc.BasicColor.GREEN if ok else ouc.BasicColor.RED
                     ).attach_to(scene)
        if ok:
            sol_robot = leaf_sampler.clone()
            sol_robot.fk(qs=qs)
            sol_robot.alpha = 0.5
            sol_robot.attach_to(scene)
            sol_robot.toggle_tcp('d405', length_scale=0.25, radius_scale=0.4,
                                 color_mat=ouc.CoordColor.DYO)
        print(f'pot {idx}/{len(points)} point={np.round(point, 4)} '
              f'ok={ok} info={info}')
        results.append((point, qs, info, ok))
    return results


def main():
    base = ovw.World(cam_pos=(1.25, 0.75, 0.65),
                     cam_lookat_pos=(0.45, -0.10, 0.12))
    ossop.frame().attach_to(base.scene)
    mobisys_base, tube_handler, leaf_sampler = mobisys()
    mobisys_base.attach_to(base.scene)
    tube_handler.attach_to(base.scene)
    leaf_sampler.attach_to(base.scene)
    leaf_sampler.toggle_tcp('d405', length_scale=0.3, radius_scale=0.5,
                            color_mat=ouc.CoordColor.MYC)

    pot_template = osso.SceneObject.from_file(
        _POT_PATH, collision_type=None, rgb=ouc.ExtendedColor.CHINESE_RED,
        alpha=0.85)
    for pot_point in pot_positions():
        pot = pot_template.clone()
        pot.set_pos_rotmat(pos=pot_point, rotmat=_POT_ROTMAT)
        pot.attach_to(base.scene)

    results = show_observations(base.scene, leaf_sampler,
                                pot_observation_points())
    reachable = sum(1 for *_, ok in results if ok)
    print(f'pots: {reachable} of {len(results)} observable')
    return base, leaf_sampler, results


if __name__ == '__main__':
    import builtins

    base, leaf_sampler, results = main()
    builtins.base = base
    builtins.leaf_sampler = leaf_sampler
    builtins.results = results
    base.run()
