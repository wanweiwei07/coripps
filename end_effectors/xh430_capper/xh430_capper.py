"""XH430-driven tube capper: a fixed base and one finger that turns on a
hinge -- opening the cap of a tube held in the base, pressing it shut.

Not carried by an arm: it stands on the platform (``robots/mobisys``), placed
there like any mechanism (``pos`` / ``rotmat`` of its frame).

Frame: the origin ON the finger's hinge axis, axes along the platform's.
The hinge axis runs (281.5, -31.75, 90.25) -> (273.5, -31.75, 90.25) mm in
the platform frame, i.e. along -x: the joint turns about -x, positive
OPENS the cap, negative closes it on the tube.

Joint angle = servo dial angle - 180 deg: the meshes were exported with the
finger at the servo's home (180 deg on its dial), so that pose is q = 0, and
the dial angles of ``exec.CONFIG`` (home 180, stopper 150, open 275, close
75, pressed past it by press_overshoot_deg) map to q directly
(:func:`q_of_dial`, :meth:`XH430Capper.set_mode`).

One tcp, ``'tube'`` (:data:`TUBE_POS`): a tube placed in it, at its cap's
top, z up its axis. For planning around it, :data:`TRANSIT_BOXES` (its meshes'
hulls fill the store's hole).

Meshes (``meshes/``): ``capper_base.stl`` in the platform frame (its 25 mm
block included), ``capper_finger.stl`` in its CAD part's own frame -- placed
by ``_FINGER_ROTMAT`` / ``_FINGER_POS``, found by matching it to the finger
in ``models/capper.stl`` (axis-aligned, < 0.6 mm).

    py -3.12 coripps/end_effectors/xh430_capper/xh430_capper.py
"""

import os

import numpy as np

import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_structure as orbms
import one.robots.base.mech_base as orbmb

from coripps.end_effectors.xh430_capper.exec import CONFIG


# the hinge, in the platform frame (the meshes'): this mechanism's origin
HINGE_POS = np.array([0.2815, -0.03175, 0.09025], dtype=np.float32)
# the tcp 'tube': a tube placed in the capper, at its cap's top on its axis,
# z up the axis (platform frame). x, y: the store's hole as the picker's
# camera found it (layout_cap.HOLE_XY); z: the hole's floor in
# capper_base.stl (28 mm, a ray down the hole) + the tube's 59 mm
# (models/tube.stl) -- where a tube let go there stands (layout_cap: 87)
TUBE_POS = np.array([0.2535, -0.04475, 0.087], dtype=np.float32)
# the capper as two boxes for planning around it ((x, y, z) min / max, m,
# this mechanism's frame -- its hinge -- as AABBs: valid while it stands
# unrotated): below 70 mm over the platform its whole footprint, above it
# 80 mm long, shortened on the -x side (the rack's); its model's top is at
# 113 mm, the boxes' 120 is the margin
_BOXES_PLATFORM = (((0.2225, -0.0825, 0.0), (0.3225, -0.0125, 0.070)),
                   ((0.2425, -0.0825, 0.070), (0.3225, -0.0125, 0.120)))
TRANSIT_BOXES = tuple((tuple(np.subtract(lo, HINGE_POS).tolist()),
                       tuple(np.subtract(hi, HINGE_POS).tolist()))
                      for lo, hi in _BOXES_PLATFORM)
# the dial angle the meshes show the finger at: q = 0
ZERO_DIAL_DEG = 180.0
# capper_finger.stl's frame -> the platform frame
_FINGER_ROTMAT = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float32)
_FINGER_POS = np.array([0.2815, 0.9217787, 0.9721498], dtype=np.float32)


def q_of_dial(dial_deg):
    """The joint angle (rad) at servo dial angle ``dial_deg``."""
    return float(np.deg2rad(dial_deg - ZERO_DIAL_DEG))


# the servo's positions (exec.CONFIG) as joint angles; 'pressed' is where a
# close drives to (past 'close': it stalls on the cap)
MODES = {
    'home': q_of_dial(CONFIG.home_deg),
    'stopper': q_of_dial(CONFIG.stopper_deg),
    'open': q_of_dial(CONFIG.open_deg),
    'close': q_of_dial(CONFIG.close_deg),
    'pressed': q_of_dial(CONFIG.close_deg - CONFIG.press_overshoot_deg),
}


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'capper_base.stl'),
        loc_pos=-HINGE_POS,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.LIGHT_SLATE_GRAY,
    )
    finger_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'capper_finger.stl'),
        loc_rotmat=_FINGER_ROTMAT,
        loc_pos=_FINGER_POS - HINGE_POS,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.BasicColor.WHITE,
    )
    jnt = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=finger_lnk,
        axis=-ouc.StandardAxis.X,
        pos=np.zeros(3, dtype=np.float32),
        lmt_lo=min(MODES.values()),
        lmt_up=max(MODES.values()),
    )
    structure.add_lnk(base_lnk)
    structure.add_lnk(finger_lnk)
    structure.add_jnt(jnt)
    structure.compile()
    return structure


class XH430Capper(orbmb.MechBase):
    """The capper on the platform: one joint, the finger's hinge angle."""

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self, rotmat=None, pos=None, mode='home'):
        super().__init__(rotmat=rotmat, pos=pos, is_floating=False)
        # on the base: it stays where it is whatever the finger does
        self.add_tcp('tube', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(pos=TUBE_POS - HINGE_POS))
        self.set_mode(mode)

    def set_mode(self, mode):
        """The finger where the real servo goes for ``mode`` (``MODES``)."""
        self.fk(qs=[MODES[mode]])

    def set_dial(self, dial_deg):
        """The finger at servo dial angle ``dial_deg`` (as read from it)."""
        self.fk(qs=[q_of_dial(dial_deg)])


if __name__ == '__main__':
    import builtins
    import pyglet.window.key as pkey
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(0.55, -0.35, 0.30),
                     cam_lookat_pos=(0.27, -0.06, 0.07))
    capper = XH430Capper(pos=HINGE_POS)
    capper.attach_to(base.scene)
    names = list(MODES)
    state = {'i': 0}
    print(f"{names[0]}: q {np.degrees(MODES[names[0]]):+.0f} deg -- space: "
          'the next mode')

    def step(dt):
        if base.input_manager.is_key_pressed_edge(pkey.SPACE):
            state['i'] = (state['i'] + 1) % len(names)
            mode = names[state['i']]
            capper.set_mode(mode)
            base.scene.dirty = True
            print(f'{mode}: q {np.degrees(MODES[mode]):+.0f} deg')

    base.schedule_interval(step, interval=0.03)
    builtins.base = base
    base.run()
