import numpy as np

import one.scene.scene_object_primitive as ossop
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.robots.manipulators.denso.cvr038 as ormdc
from coripps.end_effectors.xh430_cutter import XH430Cutter

# the cutter as assembled, off its model on the flange (m, flange frame):
# its nozzle tip seen by the stereo pair (rectified close_pair_*, tip at
# left (481, 141) / right (332, 143) px, 2026-11-19) against the model's --
# ~±0.4 mm across, ±0.7 mm along. The eyes are the hand-eye's, not moved.
# Measured (0.0055, -0.0010, 0.0022) -- but with it the nozzle came down
# further off the leaf, the other way (2026-11-19): the hand-eye likely made
# up for it, so the two go together. Left at 0 until both are checked.
CUTTER_MOUNT_FIX = (0.0, 0.0, 0.0)


class LeafSampler(ormdc.CVR038):
    """CVR038 with the XH430 four-bar cutter (its stereo cameras on it) on
    the flange."""

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        # the cutter's base origin is its flange-mating face
        self.cutter = XH430Cutter()
        loc_tf = np.eye(4, dtype=np.float32)
        loc_tf[:3, 3] = CUTTER_MOUNT_FIX
        self.mount(self.cutter, self.runtime_lnks[-1], loc_tf=loc_tf,
                   update=True)

    def clone(self):
        new = super().clone()
        new.cutter = next(c for c in new._mountings if isinstance(c, XH430Cutter))
        return new


def leaf_sampler(rotmat=None, pos=None):
    """Return a LeafSampler robot."""
    return LeafSampler(rotmat=rotmat, pos=pos)


if __name__ == '__main__':
    base = ovw.World(cam_pos=(0.8, 0.55, 0.55),
                     cam_lookat_pos=(0.0, 0.0, 0.2))
    ossop.frame().attach_to(base.scene)
    robot = LeafSampler()
    robot.attach_to(base.scene)
    robot.toggle_tcp('flange', color_mat=ouc.CoordColor.MYC)
    base.run()
