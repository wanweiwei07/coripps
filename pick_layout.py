"""Where the tube picks ARE: the candidate grid, the TCP orientations, and the
four TCP positions one pick is made of.

Definition only -- no robot, no IK, no file. It answers "which points do we
try, facing which way, and what does the approach look like", which is the
input to the solver and the thing you edit when the cell is re-arranged.

Kept in Python rather than in a config file on purpose: every number here is
paired with the reason it has that value, and JSON has nowhere to put the
reason. The SOLVED result is what belongs in a file (``pick_poses_io``).
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
# the negative-y rows (-0.06, -0.03) run along the machine, so they are
# clipped in x on top of the x<0 exclusion below
PICK_NEG_Y_X_MAX = 0.21
PICK_Z = 0.07               # tube grasp height the grid is solved at

# ---- the two approach orientations ----------------------------------------
# The default is spun -90 deg about the tool's own z; TCP_ROTMAT_NO_YAW is the
# unspun one, which the far corner points below MUST use.
TCP_ROTMAT_NO_YAW = oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi)
TCP_ROTMAT = (TCP_ROTMAT_NO_YAW @
              oum.rotmat_from_axangle(ouc.StandardAxis.Z, -np.pi / 2.0))
NO_YAW_PICK_XY = np.array([
    [0.24, 0.00],
    [0.27, 0.00],
    [0.30, 0.00],
    [0.24, 0.03],
    [0.27, 0.03],
    [0.30, 0.03],
], dtype=np.float32)

# ---- the shape of one approach --------------------------------------------
# Retreat distances along the TCP's OWN axes (columns of its rotmat), not along
# world x/z: column 0 is the lateral back-off, -column 2 the retreat along the
# approach axis, so the sequence follows whatever orientation the TCP is given.
APPROACH_BACK_DIST = 0.04
APPROACH_UP_DIST = 0.03
LIFT_DIST = 0.06


def pick_points_xy(z=PICK_Z):
    """Return TCP pick candidate positions on the configured XY grid."""
    x_values = np.arange(PICK_X_MIN, PICK_X_MAX + PICK_X_STEP * 0.5,
                         PICK_X_STEP, dtype=np.float32)
    y_values = np.arange(PICK_Y_MIN, PICK_Y_MAX + PICK_Y_STEP * 0.5,
                         PICK_Y_STEP, dtype=np.float32)
    x_eps = PICK_X_STEP * 0.5
    points = [
        (float(x), float(y), float(z))
        for y in y_values
        for x in x_values
        if not (y < 0.0 and (x < 0.0 or x > PICK_NEG_Y_X_MAX + x_eps))
    ]
    return np.asarray(points, dtype=np.float32)


def tcp_rotmats(point):
    """Return the TCP orientations to try for ``point``, best first.

    The points listed in :data:`NO_YAW_PICK_XY` MUST be picked unspun, so they
    get :data:`TCP_ROTMAT_NO_YAW` alone -- failing there is a real failure.
    Every other point prefers the unspun orientation too but may fall back on
    the spun :data:`TCP_ROTMAT`; only when both fail is the point unreachable.

    Matched with half-a-step tolerance -- the grid comes from a float32
    ``np.arange``, so its values are only approximately the nominal ones.
    """
    xy = np.asarray(point, dtype=np.float32)[:2]
    tol = min(PICK_X_STEP, PICK_Y_STEP) * 0.5
    if np.any(np.all(np.abs(NO_YAW_PICK_XY - xy) < tol, axis=1)):
        return (TCP_ROTMAT_NO_YAW,)
    return (TCP_ROTMAT_NO_YAW, TCP_ROTMAT)


def pose_points(point, tcp_rotmat):
    """Return the four TCP positions one pick is made of.

    The offsets are taken along the TCP's own axes, so the whole sequence
    rotates with ``tcp_rotmat`` instead of being pinned to world x/z:

    * ``pre_pick_high`` -- backed off along column 0 AND along -column 2,
    * ``pre_pick``      -- backed off along column 0 only,
    * ``pick``          -- the point itself,
    * ``lift``          -- pulled up along -column 2.
    """
    back = np.asarray(tcp_rotmat, dtype=np.float32)[:, 0]
    up = -np.asarray(tcp_rotmat, dtype=np.float32)[:, 2]
    return np.asarray([
        point + back * APPROACH_BACK_DIST + up * APPROACH_UP_DIST,
        point + back * APPROACH_BACK_DIST,
        point,
        point + up * LIFT_DIST,
    ], dtype=np.float32)


def is_no_yaw(tcp_rotmat):
    """True when ``tcp_rotmat`` is the unspun orientation -- for labelling."""
    return bool(np.allclose(tcp_rotmat, TCP_ROTMAT_NO_YAW))
