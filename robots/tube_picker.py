import os
import sys
import numpy as np

# Allow running as `python riken/robots/tube_picker.py` from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
import one.robots.manipulators.denso.cvr038 as ormdc

from coripps.grippers.cvr038_4tb_gripper import CVR0384TBGripper


class TubePicker(ormdc.CVR038):
    """CVR038 arm with the Riken 4TB tube-picking gripper mounted."""

    def __init__(self, rotmat=None, pos=None, jaw_width=0.03):
        super().__init__(rotmat=rotmat, pos=pos)
        self.gripper = CVR0384TBGripper()
        self.gripper.set_jaw_width(jaw_width)
        self.engage(self.gripper)

    def toggle_joint_frames(self, length_scale=0.28, radius_scale=0.35,
                            color_mat=ouc.CoordColor.RGB):
        """Toggle coordinate frames at each arm joint."""
        if getattr(self, '_joint_frames', None) is not None:
            for frame, parent_lnk in self._joint_frames:
                frame.detach_from(parent_lnk)
            self._joint_frames = None
            return None

        compiled = self.structure.compiled
        joint_ids = self._main_chain.jnt_ids_in_structure
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


def tube_picker(rotmat=None, pos=None, jaw_width=0.03):
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
    robot.toggle_tcp(color_mat=ouc.CoordColor.MYC)

    builtins.base = base
    builtins.robot = robot
    builtins.gripper = robot.gripper
    base.run()
