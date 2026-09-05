"""XM430-driven rotary (angular) two-finger gripper.

Unlike the parallel jaws elsewhere in the tree, the fingers PIVOT: each one
turns about the local -X axis through its own hinge, the left one forward and
the right one backward (a mirrored ``mmc``). The faces therefore fan open
instead of translating, so the gap depends on WHERE along the blade it is
measured -- at full travel the tip is ~46 mm apart while the root of the face
is only ~16 mm. The opening is defined at the TIP pad, which is where these
long thin blades actually grip.

Geometry (hinge frame, measured from the meshes):
    hinge at (19.25, +-12, 19.75) mm, axis -X, travel 0..15 deg
    tip pad at radius 80.62 mm from the hinge
    half-gap(q) = HINGE_Y + PAD_R * sin(q + PAD_PHI)

``_PAD_PHI`` is phased so the faces sit 4 mm apart at q = 0 (they do not close
all the way), which makes the opening invert in closed form -- ``set_opening``
is an arcsin, no sampling or fitting. Gluing rubber pads onto the faces just
subtracts twice the pad thickness: drop it out of ``_PAD_PHI`` by re-solving
``half-gap(0) = 0.002 - pad_thickness``.

The pad sits essentially at the top of its arc, so the grasp center travels
only 0.14 mm across the whole 15 deg -- a constant ``grasp_center`` tcp is
accurate here, unlike a linkage gripper whose center sweeps.
"""

import os

import numpy as np

import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_structure as orbms
import one.robots.base.mech_base as orbmb
import one.robots.end_effectors.ee_mixins as oreb


# hinge, in the gripper base frame
_HINGE_X = 0.01925
_HINGE_Y = 0.012
_HINGE_Z = 0.01975
# Tip pad, in polar form about the hinge in the Y-Z plane. Both numbers are
# DESIGN values, not mesh measurements: the pad turns on a 75 mm arm and the
# faces rest 4 mm apart at q = 0 (they never close all the way). The bare STL
# runs to 80.6 mm and tapers to 2.8 mm at its extreme tip, so the model is the
# nominal pad, not the raw blade. Rubber pads subtract twice their thickness --
# re-solve _PAD_PHI against ``_REST_HALF_GAP - pad_thickness`` once they are on.
_PAD_R = 0.075
_REST_HALF_GAP = 0.002
_PAD_PHI = float(np.arcsin((_REST_HALF_GAP - _HINGE_Y) / _PAD_R))
_TRAVEL = np.deg2rad(15.0)     # q = 0 is the rest gap, 15 deg is fully open
# grasp center: on the mount axis, at the tip-pad height
_GRASP_CENTER = (0.0, 0.0, 0.0997)


def _half_gap(q):
    """Tip-pad distance from the mid-plane at hinge angle ``q``."""
    return _HINGE_Y + _PAD_R * np.sin(q + _PAD_PHI)


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'base.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER,
    )
    # each finger mesh is drawn in the gripper base frame, so it is shifted back
    # by its own hinge offset -- at q = 0 it then lands where it was authored
    lft_fgr = orbms.Link.from_file(
        os.path.join(mesh_dir, 'lft_fgr.stl'),
        loc_pos=np.array([-_HINGE_X, -_HINGE_Y, -_HINGE_Z], dtype=np.float32),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE,
    )
    rgt_fgr = orbms.Link.from_file(
        os.path.join(mesh_dir, 'rgt_fgr.stl'),
        loc_pos=np.array([-_HINGE_X, _HINGE_Y, -_HINGE_Z], dtype=np.float32),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SALMON_PINK,
    )

    jnt_lft = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lft_fgr,
        axis=-ouc.StandardAxis.X,
        pos=np.array([_HINGE_X, _HINGE_Y, _HINGE_Z], dtype=np.float32),
        lmt_lo=0.0,
        lmt_up=_TRAVEL,
    )
    # same axis, opposite sense -- the two blades mirror about the Y mid-plane
    jnt_rgt = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=rgt_fgr,
        axis=-ouc.StandardAxis.X,
        pos=np.array([_HINGE_X, -_HINGE_Y, _HINGE_Z], dtype=np.float32),
        mmc=(jnt_lft, -1.0, 0.0),
        lmt_lo=-_TRAVEL,
        lmt_up=0.0,
    )

    structure.add_lnk(base_lnk)
    structure.add_lnk(lft_fgr)
    structure.add_lnk(rgt_fgr)
    structure.add_jnt(jnt_lft)
    structure.add_jnt(jnt_rgt)
    structure.ignore_collision(lft_fgr, rgt_fgr)
    structure.compile()
    return structure


class XM430RotaryGripper(orbmb.MechBase, oreb.GripperMixin):
    """Angular two-finger gripper. The opening is the TIP-pad gap; the hinge
    angle follows from it in closed form."""

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        super().__init__()   # is_floating=True default (free until mounted)
        self.add_tcp('grasp_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(pos=_GRASP_CENTER))
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self.jaw_range = np.array(
            [2.0 * _half_gap(0.0), 2.0 * _half_gap(_TRAVEL)], dtype=np.float32)
        self.open_dir = ouc.StandardAxis.Y
        self.set_opening(self.jaw_range[1])

    def set_opening(self, jaw_width):
        """Invert ``jaw_width = 2 * (HINGE_Y + PAD_R * sin(q + PAD_PHI))``."""
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f'jaw_width {jaw_width} out of range {self.jaw_range}')
        sin_arg = (0.5 * float(jaw_width) - _HINGE_Y) / _PAD_R
        q = np.arcsin(np.clip(sin_arg, -1.0, 1.0)) - _PAD_PHI
        self.fk(qs=[q])

    def opening(self):
        """The current tip-pad gap (the inverse of :meth:`set_opening`)."""
        return float(2.0 * _half_gap(self.qs[0]))

    def clone(self):
        new = super().clone()
        new.contact_pattern = self.contact_pattern.copy()
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir
        new.set_opening(self.opening())
        return new


if __name__ == '__main__':
    import builtins

    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=[0.3, 0.22, 0.2], cam_lookat_pos=[0, 0, 0.06])
    gripper = XM430RotaryGripper()
    gripper.fk(qs=[0.0])                       # the zero pose, solid
    gripper.attach_to(base.scene)
    ghost = XM430RotaryGripper()               # full travel, translucent
    ghost.fk(qs=[_TRAVEL])
    ghost.alpha = 0.3
    ghost.attach_to(base.scene)
    ossop.frame().attach_to(base.scene)
    gc_tf = gripper.tcp('grasp_center').tf
    ossop.frame(pos=gc_tf[:3, 3], rotmat=gc_tf[:3, :3],
                color_mat=ouc.CoordColor.MYC).attach_to(base.scene)
    print(f'q = 0 deg  -> opening {gripper.opening() * 1000:.3f} mm  (solid)')
    print(f'q = 15 deg -> opening {ghost.opening() * 1000:.3f} mm  (ghost)')
    builtins.base = base
    builtins.gripper = gripper
    base.run()
