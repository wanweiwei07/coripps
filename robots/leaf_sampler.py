import os
import sys
import numpy as np

# Allow running as `python riken/robots/leaf_sampler.py` from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.utils.constant as ouc
import one.utils.math as oum
import one.viewer.world as ovw
import one.robots.base.kine_visualizer as orbkv
import one.robots.manipulators.denso.cvr038 as ormdc


_RIKEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_D405_PATH = os.path.join(_RIKEN_DIR, 'models', 'd405.stl')
_D405_POS_J5 = np.array([-0.0245, -0.0445, 0.0235], dtype=np.float32)
_D405_ROTMAT_J5 = np.eye(3, dtype=np.float32)
_D405_TF_J5 = oum.tf_from_rotmat_pos(_D405_ROTMAT_J5, _D405_POS_J5)
_D405_FRAME_POS_D405 = np.array([-0.0361127, -0.012414, -0.0105], dtype=np.float32)
_D405_FRAME_ROTMAT_D405 = (
    oum.rotmat_from_axangle(ouc.StandardAxis.Z, -np.pi / 4.0) @
    oum.rotmat_from_axangle(ouc.StandardAxis.X, np.pi / 2.0)
).astype(np.float32)
_D405_FRAME_TF_D405 = oum.tf_from_rotmat_pos(
    _D405_FRAME_ROTMAT_D405,
    _D405_FRAME_POS_D405,
)
_D405_FRAME_TF_J5 = _D405_TF_J5 @ _D405_FRAME_TF_D405
_OBS_TARGET_POS = np.array([0.580, -0.120, 0.127], dtype=np.float32)
_OBS_TARGET_Z = np.array([0.0, 0.0, -1.0], dtype=np.float32)
_OBS_ROBOT_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, -np.pi / 2.0)
_OBS_ROBOT_POS = np.array([0.42, 0.025, 0.0], dtype=np.float32)


def solve_d405_obs_all(robot, max_solutions=8, seed_count=2048):
    # partial IK: point the d405 frame's z axis at the target, roll free.
    # 'to_j5' = the 5-joint chain (root -> j5 link); 'd405' = the camera frame.
    sols, infos = robot.ik_partial(
        'to_j5', 'd405',
        tgt_pos=_OBS_TARGET_POS,
        axis_constraints={'z': _OBS_TARGET_Z},
        max_solutions=max_solutions,
        seed_count=seed_count,
        return_infos=True,
    )
    solutions = []
    for full_qs, info in zip(sols, infos):
        if any(np.linalg.norm(full_qs - prev_qs) <= 1e-2
               for prev_qs, _ in solutions):
            continue
        solutions.append((full_qs, info))
        if len(solutions) >= max_solutions:
            break
    return solutions


class LeafSampler(ormdc.CVR038):
    """CVR038 with a D405 sensor mounted on the joint-5 link."""

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        self.d405 = osso.SceneObject.from_file(
            _D405_PATH,
            collision_type=None,
            is_free=True,
            rgb=ouc.ExtendedColor.SILVER_GRAY,
            alpha=1.0,
        )
        # mount the sensor on the j5 link
        self.mount(self.d405, self.j5_link, _D405_TF_J5, update=True)
        # 5-joint chain (root -> j5 link) + the camera frame as a tcp
        compiled = self.structure.compiled
        jidx5 = self.chain('main').jnt_ids_in_structure[4]
        j5_struct_lnk = self.structure.lnks[compiled.clidx_of_jidx[jidx5]]
        self.add_chain('to_j5', compiled.root_lnk, j5_struct_lnk)
        self.d405_frame = self.add_tcp('d405', self.j5_link, _D405_FRAME_TF_J5)

    @property
    def j5_link(self):
        """Return the runtime child link moved by joint 5."""
        jidx5 = self.chain('main').jnt_ids_in_structure[4]
        child_lidx = self.structure.compiled.clidx_of_jidx[jidx5]
        return self.runtime_lnks[child_lidx]

    def clone(self):
        new = super().clone()
        new.d405_frame = new.tcp('d405')
        return new


def leaf_sampler(rotmat=None, pos=None):
    """Return a LeafSampler robot."""
    return LeafSampler(rotmat=rotmat, pos=pos)


if __name__ == '__main__':
    import builtins

    base = ovw.World(cam_pos=(0.8, 0.55, 0.55),
                     cam_lookat_pos=(0.0, 0.0, 0.2))
    ossop.frame().attach_to(base.scene)

    robot = LeafSampler(rotmat=_OBS_ROBOT_ROTMAT, pos=_OBS_ROBOT_POS)
    sols, infos = robot.ik_partial(
        'to_j5', 'd405',
        tgt_pos=_OBS_TARGET_POS,
        axis_constraints={'z': _OBS_TARGET_Z},
        seed_count=128,
        return_infos=True,
    )
    ok = len(sols) > 0
    qs = sols[0] if ok else None
    info = infos[0] if infos else None
    if ok:
        robot.fk(qs=qs)

    robot.attach_to(base.scene)
    robot.alpha=0.3
    robot.toggle_tcp('flange', color_mat=ouc.CoordColor.MYC)
    # kv = orbkv.KineVisualizer(robot, chain=robot.chain('main'), alpha=0.8)
    # kv.attach_to(base.scene)
    ossop.frame_from_tf(
        robot.j5_link.tf,
        length_scale=0.5,
        radius_scale=0.6,
        color_mat=ouc.CoordColor.DYO,
    ).attach_to(base.scene)
    robot.toggle_tcp(
        'd405',
        length_scale=0.3,
        radius_scale=0.5,
        color_mat=ouc.CoordColor.MYC,
    )
    ossop.sphere(
        pos=_OBS_TARGET_POS,
        radius=0.006,
        segments=12,
        rgb=ouc.BasicColor.RED,
        alpha=1.0,
    ).attach_to(base.scene)
    target_rotmat = oum.rotmat_from_axis_constraints(
        {'z': _OBS_TARGET_Z}, ref_rotmat=robot.d405_frame.rotmat)
    ossop.frame_from_tf(
        oum.tf_from_rotmat_pos(target_rotmat, _OBS_TARGET_POS),
        length_scale=0.25,
        radius_scale=0.4,
        color_mat=ouc.CoordColor.RGB,
    ).attach_to(base.scene)
    print('d405 observation ik:')
    print(f'  ok={ok}')
    print(f'  info={info}')
    if qs is not None:
        d405_tf = robot.d405_frame.tf
        pos_err = float(np.linalg.norm(_OBS_TARGET_POS - d405_tf[:3, 3]))
        z_err = float(oum.axis_angle_error(d405_tf[:3, 2], _OBS_TARGET_Z))
        print(f'  qs_deg={np.round(np.degrees(qs), 3)}')
        print(f'  d405_pos={d405_tf[:3, 3]}')
        print(f'  pos_err={pos_err}')
        print(f'  z_err_deg={np.degrees(z_err)}')
    print('j5_link tf:')
    print(robot.j5_link.tf)
    print('d405_frame tf:')
    print(robot.d405_frame.tf)
    j6_link = robot.runtime_lnks[robot.structure.compiled.clidx_of_jidx[
        robot.chain('main').jnt_ids_in_structure[5]]]
    j6_in_j5_tf = oum.tf_inverse(robot.j5_link.tf) @ j6_link.tf
    print('j6 pos in j5 frame:')
    print(j6_in_j5_tf[:3, 3])

    builtins.base = base
    builtins.robot = robot
    # builtins.kv = kv
    builtins.j6_link = j6_link
    builtins.j6_in_j5_tf = j6_in_j5_tf
    builtins.d405 = robot.d405
    builtins.d405_frame = robot.d405_frame
    base.run()
