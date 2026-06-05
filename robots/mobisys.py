import os
import sys
import numpy as np

# Allow running as `python riken/robots/mobisys.py` from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.utils.constant as ouc
import one.utils.math as oum
import one.viewer.world as ovw

from coripps.robots.leaf_sampler import LeafSampler
from coripps.robots.tube_picker import TubePicker


_RIKEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_PATH = os.path.join(_RIKEN_DIR, 'models', 'base.stl')
_BASE_ROTMAT = np.eye(3, dtype=np.float32)
_BASE_POS = np.zeros(3, dtype=np.float32)
_TUBE_PICKER_ROTMAT = np.eye(3, dtype=np.float32)
_TUBE_PICKER_POS = np.zeros(3, dtype=np.float32)
_LEAF_SAMPLER_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, -np.pi / 2.0)
_LEAF_SAMPLER_POS = np.array([0.42, 0.031, 0.0], dtype=np.float32)


def mobisys():
    """Return the mobile system base model, tube picker, and leaf sampler."""
    base_model = osso.SceneObject.from_file(
        _BASE_PATH,
        collision_type=None,
        rgb=ouc.ExtendedColor.SILVER_GRAY,
        alpha=1.0,
    )
    base_model.set_pos_rotmat(pos=_BASE_POS, rotmat=_BASE_ROTMAT)

    tube_picker = TubePicker(rotmat=_TUBE_PICKER_ROTMAT, pos=_TUBE_PICKER_POS)
    leaf_sampler = LeafSampler(
        rotmat=_LEAF_SAMPLER_ROTMAT,
        pos=_LEAF_SAMPLER_POS,
    )
    return base_model, tube_picker, leaf_sampler


if __name__ == '__main__':
    import builtins

    world = ovw.World(cam_pos=(0.85, 0.55, 0.45),
                      cam_lookat_pos=(0.12, 0.05, 0.0))
    ossop.frame().attach_to(world.scene)

    base_model, robot, leaf_sampler = mobisys()
    base_model.attach_to(world.scene)
    robot.attach_to(world.scene)
    robot.gripper.attach_to(world.scene)
    robot.gripper.toggle_tcp('grasp_center', color_mat=ouc.CoordColor.MYC)
    leaf_sampler.attach_to(world.scene)
    leaf_sampler.toggle_tcp(
        'd405',
        length_scale=0.3,
        radius_scale=0.5,
        color_mat=ouc.CoordColor.MYC,
    )

    builtins.world = world
    builtins.base = world
    builtins.base_model = base_model
    builtins.robot = robot
    builtins.leaf_sampler = leaf_sampler
    world.run()
