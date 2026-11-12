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


# ---- moving between poses ------------------------------------------------
# on a move from one pose to another, nothing lower than this: the capper's
# opener, open, reaches up to near it. A move from / to lower goes straight
# up / down there first / last. Empty, grasp_center; holding a tube, the
# tube's bottom -- TUBE_BELOW_GC under grasp_center (models/tube.stl: 59 mm
# long, held by its cap's top)
TRAVEL_MIN_Z = 0.095
TUBE_BELOW_GC = 0.0592
HELD_TRAVEL_MIN_Z = TRAVEL_MIN_Z + TUBE_BELOW_GC

# ---- placing ----------------------------------------------------------------
# the store's hole; the drop point above it. The tube hangs from its cap now,
# so 12 mm higher than with the old side grip (on the tube's body, 12 mm
# below the cap's top) for the tube itself to stand where it did
# the store's hole, where a tube let go there settles (seen by the picker's
# camera alike from two hand turns 180 deg apart); the held tube is at
# grasp_center (the gripper model's, measured), so it is placed right over it
HOLE_XY = np.array([0.2535, -0.04475], dtype=np.float32)
DROP_POS = np.array([*HOLE_XY, 0.152], dtype=np.float32)
# straight down from the drop point, the tube held: it goes INTO the store
# instead of being dropped from the air (let go at 87 mm)
PLACE_DESCEND = 0.065
# after letting go, straight up this far (to 100 mm, level with PUSH_POS)
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
PUSH_POS = np.array([0.251, -0.0375, 0.1], dtype=np.float32)
# the from -x orientation: the blade line across the push
PUSH_ROTMAT = layout.TCP_ROTMAT_FROM_NX
# straight down, gripper closed: the fingertips (4.8 mm below grasp_center)
# to ~68 mm, where the old fingers' push ended -- 32 mm (4.8 deeper) tripped
# J2 / J3 collision detection at the end of the push
PUSH_DESCEND = 0.027

# ---- pop: fixed -- the lever that pries the tube back out of the capper ----
# (+(1, -3) mm, as PUSH_POS: the fingers where they were tuned)
POP_POS = np.array([0.251, -0.103, 0.100], dtype=np.float32)
POP_ROTMAT = layout.TCP_ROTMAT_FROM_NX
# straight down onto the lever until the FINGERTIPS are at this height
# (world z, absolute -- 25.2 mm did not lift the tube out)
POP_TIP_Z = 0.0215
# the fingertips are this far below grasp_center (the finger meshes reach
# 99.8 mm from the gripper base, grasp_center is at 95)
FINGERTIP_BELOW_GC = 0.0048
# the press, as the distance grasp_center travels down from POP_POS
POP_DIST = float(POP_POS[2] - (POP_TIP_Z + FINGERTIP_BELOW_GC))
# of it, the first this far is the way down to the lever (fast), the rest
# the press on it (slow) -- split on the solved path, so not an input
POP_FAST_DIST = 0.05

# ---- re-pick: fixed -- the popped tube, picked back up out of the store ----
REPICK_POS = np.array([*HOLE_XY, 0.140], dtype=np.float32)
# from above by the cap, as in the rack: the long tab is at -y now, so the
# fingers close along x -- the hand's y along world x
REPICK_ROTMAT = layout.TCP_ROTMAT_FROM_NY
# straight down to the cap, gripper open: grasp_center to the popped tube's
# cap top (~83 mm, measured), as in the rack (62 mm) -- 53 mm (87) gripped
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
