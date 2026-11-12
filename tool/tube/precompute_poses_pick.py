"""Solve the tube pick grid and write ``config/poses_pick.json``.

Headless on purpose: this is the step that PRODUCES the artifact, so it has to
run from a script or a shell without a window.

A grid point (``layout_pick``) is kept when the hand can look at its cap from
above (``pre_pick`` at LOOK_ROTMAT) AND, for every long-tab direction
(TAB_CHECK_STEP_DEG apart), grip it with one of the two grip yaws
(``layout_pick.grip_rotmat``): ``pre_pick`` turned to it, ``pick``, ``lift``
all with an IK solution in the joint limits and out of collision (the
sampler at its sim pose). Saved per point: the grasp point and its
``pre_pick`` joints; the executor solves the rest at run time.

Run it again after anything that changes what a pick means: the grid or the
heights in ``layout_pick``, the gripper geometry, or the robot mounting.

    py -3.12 coripps/tool/tube/precompute_poses_pick.py
"""

import sys
import argparse

import numpy as np

import coripps.picker_motion as pkm
import coripps.tool.tube.layout_pick as layout
import coripps.tool.tube.poses_io_pick as ppio
from coripps.robots.mobisys import mobisys


def valid_ik(picker, pln_ctx, pos, rotmat, near=None):
    """The IK solutions putting grasp_center at ``(pos, rotmat)`` inside the
    joint limits and out of collision, nearest ``near`` first (else the
    least J6 first)."""
    sols = picker.ik(np.asarray(pos, dtype=np.float32), rotmat,
                     tcp=picker.gripper.tcp('grasp_center'), max_solutions=8)
    ok = [np.asarray(q, dtype=np.float32) for q in (sols or [])
          if np.allclose(pln_ctx.enforce_bounds(q), q, atol=1e-6)
          and pln_ctx.is_state_valid(q)]
    key = ((lambda q: float(np.abs(q - near).max())) if near is not None
           else (lambda q: abs(float(q[5]))))
    return sorted(ok, key=key)


def grippable(picker, pln_ctx, point, tab_deg, look_qs):
    """Whether a cap at ``point`` with its long tab at ``tab_deg`` can be
    gripped with one of the two grip yaws."""
    up = np.array([0.0, 0.0, layout.LIFT_DIST], dtype=np.float32)
    for flip in (False, True):
        rotmat = layout.grip_rotmat(tab_deg, flip)
        turned = valid_ik(picker, pln_ctx, layout.pre_pick_pos(point),
                          rotmat, look_qs)
        if not turned:
            continue
        pick = valid_ik(picker, pln_ctx, point, rotmat, turned[0])
        if pick and valid_ik(picker, pln_ctx, point + up, rotmat, pick[0]):
            return True
    return False


def precompute(path=ppio.PICK_POSES_JSON):
    """Solve the grid and save it; returns ``(points, look_qs)`` kept."""
    _, picker, sampler = mobisys()
    pln_ctx, _ = pkm.build_planning_context(picker, sampler)
    grid = layout.pick_points_xy(z=layout.PICK_Z)
    tabs = range(0, 360, layout.TAB_CHECK_STEP_DEG)
    points, look_qs, dropped = [], [], []
    for point in grid:
        look = valid_ik(picker, pln_ctx, layout.pre_pick_pos(point),
                        layout.LOOK_ROTMAT)
        if not look:
            dropped.append((point, 'cannot look from above'))
            continue
        bad = [t for t in tabs
               if not grippable(picker, pln_ctx, point, t, look[0])]
        if bad:
            dropped.append((point, f'cannot grip at tab {bad} deg'))
            continue
        points.append(point)
        look_qs.append(look[0])
    print(f'tube pick points: total={len(grid)} kept={len(points)} '
          f'dropped={len(dropped)}')
    for point, why in dropped:
        print(f'  dropped {np.round(point[:2] * 1000).tolist()} mm: {why}')
    ppio.save(points, look_qs, path=path, pick_z=layout.PICK_Z,
              look_rise=layout.PICK_LOOK_RISE, lift_dist=layout.LIFT_DIST,
              grid_total=len(grid))
    return points, look_qs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=ppio.PICK_POSES_JSON,
                        help='where to write the JSON')
    args = parser.parse_args(argv)
    precompute(path=args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
