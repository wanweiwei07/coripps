import numpy as np

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
import one.robots.manipulators.denso.cvr038 as ormdc

from coripps.end_effectors.xh430_gripper import XH430Gripper, OPEN_TRAVEL


class TubePicker(ormdc.CVR038):
    """CVR038 arm with the XH430 rotary tube-picking gripper mounted."""

    def __init__(self, rotmat=None, pos=None, jaw_width=None):
        super().__init__(rotmat=rotmat, pos=pos)
        self.gripper = XH430Gripper()
        # default to the WORKING open angle (10 deg, ~30 mm), not the full
        # mechanical stop -- that is what the real gripper is driven to before
        # a pick, and it keeps the linkage off its limit
        self.gripper.set_opening(
            self.gripper.opening_from_angle(OPEN_TRAVEL) if jaw_width is None
            else jaw_width)
        self.mount(self.gripper, self.runtime_lnks[-1], update=True)

    def toggle_joint_frames(self, length_scale=0.28, radius_scale=0.35,
                            color_mat=ouc.CoordColor.RGB):
        """Toggle coordinate frames at each arm joint."""
        if getattr(self, '_joint_frames', None) is not None:
            for frame, parent_lnk in self._joint_frames:
                frame.detach_from(parent_lnk)
            self._joint_frames = None
            return None

        compiled = self.structure.compiled
        joint_ids = self.chain('main').jnt_ids_in_structure
        frames = []
        for jidx in joint_ids:
            parent_lidx = compiled.plidx_of_jidx[jidx]
            parent_lnk = self.runtime_lnks[parent_lidx]
            frame = ossop.frame_from_tf(
                compiled.jtf0_by_idx[jidx],
                length_scale=length_scale,
                radius_scale=radius_scale,
                color_mat=color_mat,
            )
            frame.attach_to(parent_lnk)
            frames.append((frame, parent_lnk))
        self._joint_frames = frames
        return [frame for frame, _ in frames]


def tube_picker(rotmat=None, pos=None, jaw_width=None):
    """Return a TubePicker robot."""
    return TubePicker(rotmat=rotmat, pos=pos, jaw_width=jaw_width)


if __name__ == '__main__':
    import builtins
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(0.8, 0.55, 0.55), cam_lookat_pos=(0.0, 0.0, 0.2))
    ossop.frame().attach_to(base.scene)

    robot = TubePicker()
    robot.fk(qs=np.zeros(robot.ndof, dtype=np.float32))
    robot.attach_to(base.scene)
    robot.gripper.attach_to(base.scene)
    robot.toggle_joint_frames()
    robot.gripper.toggle_tcp('grasp_center', color_mat=ouc.CoordColor.MYC)

    builtins.base = base
    builtins.robot = robot
    builtins.gripper = robot.gripper
    base.run()
