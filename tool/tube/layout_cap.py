"""Where the capping happens: the drop / push / pop / re-pick / discard poses.

Definition only -- no robot, no IK, no file, like ``layout_pick`` is for the
pick. This is the file you edit when the store, the capper, the lever or the
bin move; ``precompute_poses_cap.py`` then solves it into
``config/poses_cap.json``, and the executors read that file.

Every position is the gripper's ``grasp_center`` in the world frame of the
cell model (the same frame as the pick grid), in metres. The tube is held
from above by its cap (``layout_pick``), so grasp_center is at the cap's top.

* The PLACING (drop point, descent, release): the tube is carried there
  turned so its cap's long tab points to -y. The hand's x lies along the long
  tab-strap line, towards the long tab or away from it as it was gripped
  (``layout_pick.grip_rotmat``), so it arrives in one of PLACINGS.
* Everything after it -- PUSH, POP, RE-PICK of the popped tube, DISCARD -- is
  a FIXED pose of the cell, the same for every tube, each with its own
  position and TCP orientation below.
"""

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import coripps.tool.tube.layout_pick as layout
from coripps.robots.mobisys import capper_tube_pos


# ---- moving between poses ------------------------------------------------
# a move from one pose to another goes straight up from where the arm is
# until it is clear of the transit obstacles (mobisys.platform_obstacles: a
# plane over the rack's tubes, the capper as boxes) by TRAVEL_MARGIN more,
# across (checked against them), and straight down to the goal the same
# way -- each end only as high as it has to: TRAVEL_RISE_MAX at most
# (higher only for a straight across, see TRAVEL_LIFT_STEP).
# Holding a tube, the tube counts too: TUBE_BELOW_GC long under
# grasp_center along the hand's axis (models/tube.stl: 59 mm, held by its
# cap's top), TUBE_RADIUS round it
TRAVEL_MARGIN = 0.005
TRAVEL_RISE_MAX = 0.150
# when the straight joint line between the two ends is not free, both ends
# go up by TRAVEL_LIFT_STEP more and the line is tried again, up to
# TRAVEL_LIFT_MAX more; RRT-Connect between the low ends only when none is
TRAVEL_LIFT_STEP = 0.020
TRAVEL_LIFT_MAX = 0.100
TUBE_BELOW_GC = 0.0592
TUBE_RADIUS = 0.013

# ---- placing ----------------------------------------------------------------
# the store's hole; the drop point above it. The tube hangs from its cap now,
# so 12 mm higher than with the old side grip (on the tube's body, 12 mm
# below the cap's top) for the tube itself to stand where it did
# the store's hole, where a tube let go there settles (seen by the picker's
# camera alike from two hand turns 180 deg apart); the held tube is at
# grasp_center (the gripper model's, measured), so it is placed right over it
# (the capper's 'tube' tcp, mobisys.capper_tube_pos: the camera's
# (253.5, -44.75) mm, a tube home in the store, its cap's top, at 87 mm).
# Every pose ON the capper below is that tcp + an offset (world axes: the
# capper stands unrotated), so they move with it. The offsets are the poses
# tuned on the robot before the capper went up 25 mm on its block (2026-11),
# when that cap top was at 62 mm
TUBE_TCP = capper_tube_pos()


def on_capper(dx, dy, dz):
    """The capper's 'tube' tcp + (dx, dy, dz) m, world axes."""
    return (TUBE_TCP + np.array([dx, dy, dz], dtype=np.float32)).astype(
        np.float32)


HOLE_XY = TUBE_TCP[:2]
DROP_POS = on_capper(0.0, 0.0, 0.090)       # 177 mm
# straight down from the drop point, the tube held: it goes INTO the store
# instead of being dropped from the air (let go at 112 mm, its cap 25 over
# home: the push seats it)
PLACE_DESCEND = 0.065
# after letting go, straight up this far (to 125 mm, level with PUSH_POS)
RELEASE_RISE = 0.013
# the hand's orientation arriving at the drop point, the long tab at -y:
# its x towards the long tab = -y (the grip 'towards') or away = +y
PLACINGS = {
    'towards': (layout.TCP_ROTMAT_FROM_NY, DROP_POS),
    'away': ((oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi)
              @ layout.TCP_ROTMAT_FROM_NY).astype(np.float32), DROP_POS),
}


def placing_of(flip):
    """The PLACINGS key of a tube gripped with ``layout_pick.grip_rotmat``'s
    ``flip``."""
    return 'away' if flip else 'towards'


# ---- push: fixed ------------------------------------------------------------
# starts over the BACK of the placed tube, just above it
# (tuned on the robot with grasp_center at the gripper's axis: +(1, -3) mm
# keeps the fingers where they were, grasp_center now measured off the axis)
PUSH_POS = on_capper(-0.0025, 0.00725, 0.038)     # (251, -37.5, 125) mm
# the from -x orientation: the blade line across the push
PUSH_ROTMAT = layout.TCP_ROTMAT_FROM_NX
# straight down, gripper closed: the fingertips (4.8 mm below grasp_center)
# to ~93 mm (68 before the capper went up), where the old fingers' push
# ended -- 32 mm (4.8 deeper) tripped
# J2 / J3 collision detection at the end of the push
PUSH_DESCEND = 0.027

# ---- pop: fixed -- the lever that pries the tube back out of the capper ----
# (+(1, -3) mm, as PUSH_POS: the fingers where they were tuned)
POP_POS = on_capper(-0.0025, -0.05825, 0.038)     # (251, -103, 125) mm
POP_ROTMAT = layout.TCP_ROTMAT_FROM_NX
# straight down onto the lever until the FINGERTIPS are at this height
# (25.2 mm -- 50.2 now -- did not lift the tube out; 46.5 lifted it but the
# lever bottomed out first: J2 collision detection at 47.2, 2026-11-17)
POP_TIP_Z = float(TUBE_TCP[2]) - 0.039         # 48.0 mm
# the fingertips are this far below grasp_center (the finger meshes reach
# 99.8 mm from the gripper base, grasp_center is at 95)
FINGERTIP_BELOW_GC = 0.0048
# the press, as the distance grasp_center travels down from POP_POS
POP_DIST = float(POP_POS[2] - (POP_TIP_Z + FINGERTIP_BELOW_GC))
# of it, the first this far is the way down to the lever (fast), the rest
# the press on it (slow) -- split on the solved path, so not an input
POP_FAST_DIST = 0.05

# ---- re-pick: fixed -- the popped tube, picked back up out of the store ----
REPICK_POS = on_capper(0.0, 0.0, 0.078)         # 165 mm
# from above by the cap, as in the rack: the long tab is at -y now, so the
# fingers close along x -- the hand's y along world x
REPICK_ROTMAT = layout.TCP_ROTMAT_FROM_NY
# straight down to the cap, gripper open: grasp_center to the popped tube's
# cap top (~108 mm: 83 measured before the capper went up), as in the
# rack (62 mm) -- 53 mm gripped
# above it and missed
REPICK_DESCEND = 0.057

# ---- discard: fixed -- the bin the popped tube is dropped into -------------
DISCARD_POS = np.array([0.0, -0.150, 0.150], dtype=np.float32)
# re-picked from -y, turned -70 about +z on the way: J6 stays where the
# re-pick left it (-91 -> -95; -163, near its limit, with no turn) -- the
# tube drops into the bin whichever way it points
DISCARD_ROTMAT = (oum.rotmat_from_axangle(ouc.StandardAxis.Z,
                                          np.radians(-70.0))
                  @ layout.TCP_ROTMAT_FROM_NY).astype(np.float32)


def inputs():
    """Every number the solved file depends on -- stored in it, so a file
    solved from different values is refused instead of silently run."""
    as_list = lambda a: np.asarray(a, dtype=np.float32).tolist()
    return {
        'drop_pos': as_list(DROP_POS),
        **{f'placing_rotmat_{name}': as_list(rotmat)
           for name, (rotmat, _) in PLACINGS.items()},
        'place_descend': float(PLACE_DESCEND),
        'release_rise': float(RELEASE_RISE),
        'push_pos': as_list(PUSH_POS),
        'push_rotmat': as_list(PUSH_ROTMAT),
        'push_descend': float(PUSH_DESCEND),
        'pop_pos': as_list(POP_POS),
        'pop_rotmat': as_list(POP_ROTMAT),
        'pop_dist': float(POP_DIST),
        'repick_pos': as_list(REPICK_POS),
        'repick_rotmat': as_list(REPICK_ROTMAT),
        'repick_descend': float(REPICK_DESCEND),
        'discard_pos': as_list(DISCARD_POS),
        'discard_rotmat': as_list(DISCARD_ROTMAT),
        'path_step': float(layout.PATH_STEP),
    }
