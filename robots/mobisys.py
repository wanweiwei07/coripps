import os
import numpy as np

import one.geom.geometry as ogg
import one.geom.loader as ogl
import one.scene.render_model as osrm
import one.scene.scene as oss
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.utils.constant as ouc
import one.utils.math as oum
import one.viewer.world as ovw

from coripps.robots.leaf_sampler import LeafSampler
from coripps.robots.tube_picker import TubePicker
from coripps.end_effectors.xh430_capper.xh430_capper import (
    XH430Capper, HINGE_POS, TUBE_POS, TRANSIT_BOXES)


_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
# the platform, split: the cart, the tube rack on it -- in the same world
# frame -- and the capper (raised 25 mm on its block), a mechanism of its own
# (end_effectors/xh430_capper: its finger turns), at its hinge. The cart is a box:
# models/base.stl's bounding box (its plate on top, z 0..5 mm, the casters
# inside) -- the mesh was 635k triangles for a frame
_CART_MIN = np.array([-0.250, -0.078, -0.379], dtype=np.float32)
_CART_MAX = np.array([0.516, 0.253, 0.005], dtype=np.float32)
_CART_RGB = ouc.ExtendedColor.SILVER_GRAY
_BASE_PARTS = (
    ('tube_rack.stl', ouc.BasicColor.WHITE),    # white PLA
)
_BASE_ROTMAT = np.eye(3, dtype=np.float32)
_BASE_POS = np.zeros(3, dtype=np.float32)
_TUBE_PICKER_ROTMAT = np.eye(3, dtype=np.float32)
_TUBE_PICKER_POS = np.array([0.0, 0.0, 0.005], dtype=np.float32)
_LEAF_SAMPLER_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, -np.pi / 2.0)
_LEAF_SAMPLER_POS = np.array([0.42, 0.031, 0.005], dtype=np.float32)
# the capper's frame (its hinge) on the platform
_CAPPER_POS = (_BASE_POS + _BASE_ROTMAT @ HINGE_POS).astype(np.float32)


CAPPER_BOXES = tuple(
    (tuple((_CAPPER_POS + np.asarray(lo)).tolist()),
     tuple((_CAPPER_POS + np.asarray(hi)).tolist()))
    for lo, hi in TRANSIT_BOXES)


def capper_tube_pos():
    """The capper's 'tube' tcp in the world: a tube placed in it, its cap's
    top on its axis -- as ``mobisys()[0].capper.tcp('tube')``, without
    building the scene."""
    return (_CAPPER_POS + _BASE_ROTMAT @ (TUBE_POS - HINGE_POS)).astype(
        np.float32)

# ---- the platform as the picker's transit obstacles -----------------------
# for the picker's moves between places (the across of a transfer), not for
# its straight moves into the rack or the capper: a plane over the tubes in
# the rack (their caps' tops at 62 mm) and the capper as its TRANSIT_BOXES,
# where it stands ((x, y, z) min / max, m, world)
PLATFORM_PLANE_Z = 0.070


class Platform(osso.SceneObject):
    """The platform's static parts (the cart, the tube rack) as one object,
    and its capper -- ``capper``, a mechanism -- going with it into and out
    of a scene."""

    capper = None

    def attach_to(self, target):
        super().attach_to(target)
        if self.capper is not None and isinstance(target, oss.Scene):
            self.capper.attach_to(target)

    def detach_from(self, target):
        super().detach_from(target)
        if self.capper is not None and isinstance(target, oss.Scene):
            self.capper.detach_from(target)


def mobisys():
    """Return the mobile system base model (cart + tube rack, its capper as
    ``.capper``), tube picker, and leaf sampler."""
    base_model = Platform(collision_type=None)
    base_model.add_visual(
        osrm.RenderModel(geom=ogg.gen_box_geom(_CART_MAX - _CART_MIN),
                         pos=(_CART_MIN + _CART_MAX) / 2, rgb=_CART_RGB,
                         alpha=1.0),
        auto_make_collision=False)
    for name, rgb in _BASE_PARTS:
        base_model.add_visual(
            osrm.RenderModel(geom=ogl.load_geometry(
                os.path.join(_MODEL_DIR, name)), rgb=rgb, alpha=1.0),
            auto_make_collision=False)
    base_model.set_pos_rotmat(pos=_BASE_POS, rotmat=_BASE_ROTMAT)
    base_model.capper = XH430Capper(pos=_CAPPER_POS, rotmat=_BASE_ROTMAT)

    tube_picker = TubePicker(rotmat=_TUBE_PICKER_ROTMAT, pos=_TUBE_PICKER_POS)
    leaf_sampler = LeafSampler(
        rotmat=_LEAF_SAMPLER_ROTMAT,
        pos=_LEAF_SAMPLER_POS,
    )
    return base_model, tube_picker, leaf_sampler


def platform_obstacles():
    """:data:`PLATFORM_PLANE_Z`'s plane and :data:`CAPPER_BOXES`, as scene
    objects with collision."""
    out = [ossop.plane(pos=(0.0, 0.0, PLATFORM_PLANE_Z))]
    for lo, hi in CAPPER_BOXES:
        lo, hi = np.asarray(lo), np.asarray(hi)
        out.append(ossop.box(pos=(lo + hi) / 2, xyz_lengths=hi - lo,
                             collision_type=ouc.CollisionType.AABB))
    return out


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

    builtins.world = world
    builtins.base = world
    builtins.base_model = base_model
    builtins.robot = robot
    builtins.leaf_sampler = leaf_sampler
    world.run()
