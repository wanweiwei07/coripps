"""Where the tube picks ARE: the candidate grid, and how one pick goes.

Definition only -- no robot, no IK, no file. It answers "which points do we
try, and what does a pick look like", which is the input to the solver and
the thing you edit when the cell is re-arranged.

The gripper's straight fingers take the tube from ABOVE, by the rim of its
cap, pinching the two sides across the line from the cap's long tab to its
strap (the fingers close along the gripper's y, so its x lies along that
line). One pick:

* ``pre_pick`` -- PICK_LOOK_RISE above the cap, the hand at LOOK_ROTMAT
  (from -y): the camera looks at the cap from here (the long tab's
  direction, ``vision/picker_rgb/inrack_cap_orientation``);
* the hand turned there, on the spot, to the grip yaw: :func:`grip_rotmat`
  of the long tab's direction -- x towards the long tab, or the other way
  (J6 reaches only one of the two for some directions);
* ``pick`` -- straight down to the point; ``lift`` -- straight up LIFT_DIST.

The grip yaw depends on the tube, so the moves are solved at run time; the
solver (``precompute_poses_pick``) only keeps the points every tab direction
can be picked at, with the joints of their ``pre_pick``.

Kept in Python rather than in a config file on purpose: every number here is
paired with the reason it has that value, and JSON has nowhere to put the
reason. The SOLVED result is what belongs in a file (``poses_io_pick``).
"""

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc


# ---- the candidate grid ---------------------------------------------------
PICK_X_MIN = -0.21
PICK_X_MAX = 0.30
PICK_X_STEP = 0.03
PICK_Y_MIN = -0.06
PICK_Y_MAX = 0.24
PICK_Y_STEP = 0.03
# corners of the grid with no picks, ((x_min, x_max), (y_min, y_max)) in m;
# the edges sit half way between grid lines
PICK_EXCLUDED = (
    ((-np.inf, 0.105), (-np.inf, 0.105)),
    # the capper / store (layout_cap, x ~0.25, y -0.10..-0.03) and the row
    # right behind it
    ((0.225, np.inf), (-np.inf, 0.015)),
)
# the grasp: at the cap's top (a tube in the rack, 62 mm), grasp_center on
# the gripper's axis
PICK_Z = 0.062

# ---- the hand's orientations ---------------------------------------------
# TCP pointing down, its x = world -x
TCP_ROTMAT_FROM_NX = oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi)
# that turned -90 deg about the tool's z, its x = world -y
TCP_ROTMAT_FROM_NY = (TCP_ROTMAT_FROM_NX @
                      oum.rotmat_from_axangle(ouc.StandardAxis.Z,
                                              -np.pi / 2.0))
# looking at a cap: always this way, whatever the tube
LOOK_ROTMAT = TCP_ROTMAT_FROM_NY

# ---- the shape of one pick -----------------------------------------------
# pre_pick this far above the grasp: the camera ~46 mm above the cap (the
# lens is focused near), the fingertips ~10 mm clear of it
PICK_LOOK_RISE = 0.015
# then straight up this far with the tube: out of its hole in the rack (the
# move on keeps a held tube's bottom high: layout_cap.HELD_TRAVEL_MIN_Z)
LIFT_DIST = 0.065
# the fingers' opening on the way down to the cap: just wider than the cap
# (~22 mm across them), to keep clear of the neighbours -- a neighbour's
# long tab pointing at a finger reaches 16.5 mm from its axis, a finger's
# outer face ~17.4 mm out at 26.5 (~19.3 at the working open 30), by the
# model
PICK_OPEN_WIDTH = 0.0265
# a straight TCP line is sampled this finely (a joint-space move between
# its ends bows sideways by mm)
PATH_STEP = 0.005
# the tab directions every kept point has to be pickable at (deg apart)
TAB_CHECK_STEP_DEG = 10


def pick_points_xy(z=PICK_Z):
    """Return TCP pick candidate positions on the configured XY grid."""
    x_values = np.arange(PICK_X_MIN, PICK_X_MAX + PICK_X_STEP * 0.5,
                         PICK_X_STEP, dtype=np.float32)
    y_values = np.arange(PICK_Y_MIN, PICK_Y_MAX + PICK_Y_STEP * 0.5,
                         PICK_Y_STEP, dtype=np.float32)
    points = [
        (float(x), float(y), float(z))
        for y in y_values
        for x in x_values
        if not any(xa < x < xb and ya < y < yb
                   for (xa, xb), (ya, yb) in PICK_EXCLUDED)
    ]
    return np.asarray(points, dtype=np.float32)


def grip_rotmat(tab_deg, flip=False):
    """The hand gripping a cap whose long tab points to ``tab_deg`` (world,
    deg, counter-clockwise > 0 seen from above, 0 = +x): pointing down, its
    x towards the long tab -- ``flip``: away from it -- so the fingers close
    across the long tab-strap line."""
    yaw = np.radians(tab_deg + (180.0 if flip else 0.0) - 270.0)
    return (oum.rotmat_from_axangle(ouc.StandardAxis.Z, yaw)
            @ TCP_ROTMAT_FROM_NY).astype(np.float32)


def pre_pick_pos(point):
    return np.asarray(point, dtype=np.float32) + np.array(
        [0.0, 0.0, PICK_LOOK_RISE], dtype=np.float32)
