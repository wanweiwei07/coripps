"""Solve the whole tube pick grid and write ``config/poses_pick.json``.

Headless on purpose: this is the step that PRODUCES the artifact, so it has to
run from a script or a shell without a window. Use ``pick_poses_browser.py`` to
look at what came out -- it reads the file instead of re-solving it.

Run it again after anything that changes what a pick means: the grid or the
approach in ``layout_pick``, the gripper geometry, or the robot mounting.

    py -3.12 coripps/tool/precompute_poses_pick.py
    py -3.12 coripps/tool/precompute_poses_pick.py --quiet   # skip diagnostics
"""

import sys
import argparse

import coripps.tool.layout_pick as layout
import coripps.tool.poses_io_pick as ppio
import coripps.tool.pose_sequence as pseq
from coripps.robots.mobisys import mobisys


def precompute(quiet=False, path=ppio.PICK_POSES_JSON):
    """Solve the grid and save it; returns what was saved."""
    _, tube_handler, _ = mobisys()
    points = layout.pick_points_xy(z=layout.PICK_Z)
    if not quiet:
        pseq.print_diagnostics(tube_handler, points, layout.tcp_rotmats,
                               layout.pose_path, ppio.POSE_NAMES)
    valid, paths, marks, rotmats, invalid = pseq.filter_valid(
        tube_handler, points, layout.tcp_rotmats, layout.pose_path)
    print(f'tube pick points: total={len(points)} valid={len(valid)} '
          f'invalid={len(invalid)}  ({paths.shape[1]} straight-line samples '
          f'per pick, key poses at {list(marks)})')
    ppio.save(valid, paths, marks, rotmats, path=path,
              pick_z=layout.PICK_Z, grid_total=len(points),
              path_step=layout.PATH_STEP)
    return valid, paths, marks, rotmats, invalid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quiet', action='store_true',
                        help='skip the per-pose / per-column failure counts')
    parser.add_argument('--out', default=ppio.PICK_POSES_JSON,
                        help='where to write the JSON')
    args = parser.parse_args(argv)
    precompute(quiet=args.quiet, path=args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
