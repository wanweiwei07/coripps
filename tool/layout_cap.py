"""Where the capping happens: the drop / push / pop / re-pick / discard poses.

Definition only -- no robot, no IK, no file, like ``layout_pick`` is for the
pick. This is the file you edit when the store, the capper, the lever or the
bin move; ``precompute_poses_cap.py`` then solves it into
``config/poses_cap.json``, and the executors read that file.

Every position is the gripper's ``grasp_center`` in the world frame of the
cell model (the same frame as the pick grid), in metres.

* The PLACING (drop point, descent, release) follows the tube's pick
  orientation: a tube hangs vertical in the blades only in the orientation it
  was picked with, and the two pick orientations (``layout_pick``) hold it
  slightly differently.
* Everything after it -- PUSH, POP, RE-PICK of the popped tube, DISCARD -- is
  a FIXED pose of the cell, the same for every tube, each with its own
  position and TCP orientation below.
"""

import numpy as np

import coripps.tool.layout_pick as layout


# ---- placing: per pick orientation ----------------------------------------
# a tube picked spun (layout_pick.TCP_ROTMAT)
DROP_POS = np.array([0.250, -0.0445, 0.140], dtype=np.float32)
# a tube picked unspun (layout_pick.TCP_ROTMAT_NO_YAW): only x differs
DROP_POS_NO_YAW = np.array([0.252, DROP_POS[1], DROP_POS[2]],
                           dtype=np.float32)
# straight down from the drop point, the tube held: it goes INTO the store
# instead of being dropped from the air
PLACE_DESCEND = 0.065
# after letting go of the placed tube, first slide this far along the TCP's
# own +x (in the drop orientation: the way the gripper backed off the tube
# on the pick approach) -- the open blades leave the tube sideways --
RELEASE_SHIFT = 0.025
# -- then rise only this far (not back up to the drop point) before moving
# on to the push
RELEASE_RISE = 0.030

# ---- push: fixed ------------------------------------------------------------
# starts over the BACK of the placed tube, just above it
PUSH_POS = np.array([0.250, -0.0345, 0.095], dtype=np.float32)
# the unspun orientation: the blade line across the push
PUSH_ROTMAT = layout.TCP_ROTMAT_NO_YAW
PUSH_DESCEND = 0.027            # straight down, gripper closed

# ---- pop: fixed -- the lever that pries the tube back out of the capper ----
POP_POS = np.array([0.250, -0.100, 0.100], dtype=np.float32)
POP_ROTMAT = layout.TCP_ROTMAT_NO_YAW
POP_DIST = 0.0775               # straight down onto the lever

# ---- re-pick: fixed -- the popped tube, picked back up out of the store ----
REPICK_POS = np.array([0.250, -0.0445, 0.140], dtype=np.float32)
REPICK_ROTMAT = layout.TCP_ROTMAT
REPICK_DESCEND = 0.065          # straight down to it, gripper open

# ---- discard: fixed -- the bin the popped tube is dropped into -------------
DISCARD_POS = np.array([0.0, 0.320, 0.150], dtype=np.float32)
DISCARD_ROTMAT = layout.TCP_ROTMAT  # carried as it was re-picked

# pick orientation name -> (TCP orientation, drop point)
ORIENTATIONS = {
    'spun': (layout.TCP_ROTMAT, DROP_POS),
    'no_yaw': (layout.TCP_ROTMAT_NO_YAW, DROP_POS_NO_YAW),
}


def orientation_of(tcp_rotmat):
    """The :data:`ORIENTATIONS` key a tube's pick orientation belongs to."""
    return 'no_yaw' if layout.is_no_yaw(tcp_rotmat) else 'spun'


def inputs():
    """Every number the solved file depends on -- stored in it, so a file
    solved from different values is refused instead of silently run."""
    as_list = lambda a: np.asarray(a, dtype=np.float32).tolist()
    return {
        'drop_pos': as_list(DROP_POS),
        'drop_pos_no_yaw': as_list(DROP_POS_NO_YAW),
        'place_descend': float(PLACE_DESCEND),
        'release_shift': float(RELEASE_SHIFT),
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
