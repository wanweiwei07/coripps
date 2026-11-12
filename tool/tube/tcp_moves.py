"""grasp_center moves solved by IK at run time: straight lines, and turns
about a vertical axis (on the spot by default).

A library, nothing to run. :class:`TcpMoves` works on a
:class:`~coripps.tool.tube.pick_sequence_single.PickSession` (its picker
model and planning context): every move is sampled -- lines LINE_STEP,
turns TURN_STEP_DEG apart -- each sample solved next to the one before it
and checked (joint limits, collision, no jump to another IK branch), so the
executor can play the samples as a straight move.

Angles are in the world, degrees, counter-clockwise > 0 seen from above
(world +z).
"""

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc


TURN_STEP_DEG = 1.0             # turns: IK samples this far apart
MIN_TURN_DEG = 3.0              # a partial turn shorter than this fails
LINE_STEP = 0.005               # m, straight lines: IK samples this far apart
MAX_JOINT_STEP_DEG = 5.0        # a sample further than this in a joint: a
                                # jump to another IK branch, refused


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


class TcpMoves:
    """Line and turn paths for ``session``'s picker (its ``picker``,
    ``pln_ctx``)."""

    def __init__(self, session):
        self.s = session
        self.tcp = session.picker.gripper.tcp('grasp_center')

    def tcp_pose(self, qs):
        """grasp_center ``(pos, rotmat)`` at joints ``qs``."""
        self.s.picker.fk(qs=qs)
        return (np.array(self.tcp.tf[:3, 3]),
                np.array(self.tcp.tf[:3, :3]))

    def _extend(self, path, pos, rotmat):
        """The IK sample next to ``path[-1]`` reaching ``(pos, rotmat)``,
        appended; False when there is none (no IK, a jump, a collision).
        The IK gets a copy of ``ref_qs``: it may write into it, and the
        path's first config is the caller's start."""
        sols = self.s.picker.ik(pos, rotmat, tcp=self.tcp,
                                ref_qs=np.array(path[-1]))
        if not sols:
            return False
        q = min(sols, key=lambda q: np.abs(q - path[-1]).max()).astype(
            np.float32)
        if (np.degrees(np.abs(q - path[-1]).max()) > MAX_JOINT_STEP_DEG
                or not self.s.pln_ctx.is_state_valid(q)):
            return False
        path.append(q)
        return True

    def line_path(self, start, delta):
        """grasp_center from joints ``start`` by ``delta`` (m, world) in a
        straight line, the orientation kept; None when it cannot."""
        pos, rotmat = self.tcp_pose(start)
        delta = np.asarray(delta, dtype=np.float64)
        n = max(1, int(np.ceil(np.linalg.norm(delta) / LINE_STEP)))
        path = [np.array(start, dtype=np.float32)]    # a copy: see _extend
        for k in range(1, n + 1):
            if not self._extend(path, pos + delta * k / n, rotmat):
                print(f'  the line stops {k - 1}/{n} of the way')
                return None
        return path

    def arc_path(self, start, deg, axis_xy=None, partial=False):
        """grasp_center (pose and place) from joints ``start`` turned by
        ``deg`` about the vertical world axis through ``axis_xy`` (default:
        grasp_center itself -- a turn on the spot), TURN_STEP_DEG apart:
        ``(path, got)``. ``partial``: as far as the arm can, ``(None, 0)``
        when it stops under MIN_TURN_DEG (all of a smaller ``deg`` is
        fine); else ``(None, 0)`` unless all of ``deg``."""
        pos, rotmat = self.tcp_pose(start)
        axis = pos.copy() if axis_xy is None else np.array(
            [axis_xy[0], axis_xy[1], pos[2]])
        n = max(1, int(np.ceil(abs(deg) / TURN_STEP_DEG)))
        path = [np.array(start, dtype=np.float32)]    # a copy: see _extend
        for k in range(1, n + 1):
            turn = oum.rotmat_from_axangle(ouc.StandardAxis.Z,
                                           np.radians(deg * k / n))
            if not self._extend(path, axis + turn @ (pos - axis),
                                turn @ rotmat):
                print(f'  the arm stops {deg * (k - 1) / n:+.1f} deg into '
                      'the turn (reach / joint limit / collision)')
                break
        got = deg * (len(path) - 1) / n
        short = len(path) - 1 < n       # the arm stopped before all of deg
        if short and (not partial or abs(got) < MIN_TURN_DEG):
            return None, 0.0
        return path, got
