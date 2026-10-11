"""Solve the whole tube pick grid and write ``config/pick_poses.json``.

Headless on purpose: this is the step that PRODUCES the artifact, so it has to
run from a script or a shell without a window. Use ``pick_poses_browser.py`` to
look at what came out -- it reads the file instead of re-solving it.

Run it again after anything that changes what a pick means: the grid or the
approach in ``pick_layout``, the gripper geometry, or the robot mounting.

    py -3.12 coripps/tool/precompute_pick_poses.py
    py -3.12 coripps/tool/precompute_pick_poses.py --quiet   # skip diagnostics
"""

import os
import sys
import argparse

# Allow running as `py -3.12 coripps/tool/precompute_pick_poses.py` from root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import coripps.pick_layout as layout
import coripps.pose_sequence as pseq
import coripps.pick_poses_io as ppio
from coripps.robots.mobisys import mobisys


def precompute(quiet=False, path=ppio.PICK_POSES_JSON):
    """Solve the grid and save it; returns what was saved."""
    _, tube_handler, _ = mobisys()
    points = layout.pick_points_xy(z=layout.PICK_Z)
    if not quiet:
        pseq.print_diagnostics(tube_handler, points, layout.tcp_rotmats,
                               layout.pose_points, ppio.POSE_NAMES)
    valid, sequences, rotmats, invalid = pseq.filter_valid(
        tube_handler, points, layout.tcp_rotmats, layout.pose_points)
    print(f'tube pick points: total={len(points)} valid={len(valid)} '
          f'invalid={len(invalid)}')
    ppio.save(valid, sequences, rotmats, path=path,
              pick_z=layout.PICK_Z, grid_total=len(points))
    return valid, sequences, rotmats, invalid


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
