import os
import numpy as np

import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_structure as orbms
import one.robots.base.mech_base as orbmb
import one.robots.end_effectors.ee_mixins as oreb


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'gripper_base.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER,
    )
    lf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'left_finger_4tb.stl'),
        loc_pos=np.array([0.0, 0.0, -0.06], dtype=np.float32),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE,
    )
    rf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'right_finger_4tb.stl'),
        loc_pos=np.array([0.0, 0.0, -0.06], dtype=np.float32),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SALMON_PINK,
    )

    jnt_lf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk,
        child_lnk=lf_lnk,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0.0, 0.0, 0.06], dtype=np.float32),
        lmt_lo=0.0,
        lmt_up=0.015,
    )
    jnt_rf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk,
        child_lnk=rf_lnk,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0.0, 0.0, 0.06], dtype=np.float32),
        mmc=(jnt_lf, -1.0, 0.0),
        lmt_lo=-0.015,
        lmt_up=0.0,
    )

    structure.add_lnk(base_lnk)
    structure.add_lnk(lf_lnk)
    structure.add_lnk(rf_lnk)
    structure.add_jnt(jnt_lf)
    structure.add_jnt(jnt_rf)
    structure.ignore_collision(lf_lnk, rf_lnk)
    structure.compile()
    return structure


class CVR0384TBGripper(orbmb.MechBase, oreb.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        super().__init__()   # is_free=True default
        self.add_tcp('grasp_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(pos=(0.0, 0.0, 0.1045)))
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self.jaw_range = np.array([0.0, 0.03], dtype=np.float32)
        self.open_dir = ouc.StandardAxis.Y
        self.set_opening(self.jaw_range[1])

    def set_opening(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f'jaw_width {jaw_width} out of range {self.jaw_range}')
        self.fk(qs=[jaw_width * 0.5])

    def clone(self):
        new = super().clone()
        new.contact_pattern = self.contact_pattern.copy()
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir
        new.set_opening(self.qs[0] * 2.0)
        return new


if __name__ == '__main__':
    import builtins

    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=[0.35, 0.25, 0.25], cam_lookat_pos=[0, 0, 0.07])
    gripper = CVR0384TBGripper()
    gripper.attach_to(base.scene)
    gripper.close()
    ossop.frame().attach_to(base.scene)
    gc_tf = gripper.tcp('grasp_center').tf
    ossop.frame(
        pos=gc_tf[:3, 3],
        rotmat=gc_tf[:3, :3],
        color_mat=ouc.CoordColor.MYC,
    ).attach_to(base.scene)
    builtins.base = base
    builtins.gripper = gripper
    base.run()
