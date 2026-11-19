import one.scene.scene_object_primitive as ossop
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.robots.manipulators.denso.cvr038 as ormdc
from coripps.end_effectors.xh430_cutter import XH430Cutter


class LeafSampler(ormdc.CVR038):
    """CVR038 with the XH430 four-bar cutter (its stereo cameras on it) on
    the flange."""

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        # the cutter's base origin is its flange-mating face
        self.cutter = XH430Cutter()
        self.mount(self.cutter, self.runtime_lnks[-1], update=True)

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
