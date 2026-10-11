"""The picker runs the PRECOMPUTED pick poses instead of solving them again.

Same tour as ``picker_motion``: preview the move as an animation, then send
it to the controller, one tube per ``space``. The only difference is where the
tubes come from -- ``config/pick_poses.json``, written by
``precompute_pick_poses`` -- so startup is instant and the arm runs exactly the
poses that were reviewed in the viewer, not a fresh IK solution that may have
picked a different branch.

The file is validated against the live model before anything moves: the joint
count, the pose names and the grasp height all have to match, otherwise the
stored angles do not describe this robot and running them would be unsafe.

Regenerate the file after ANY change to the grid, the TCP orientations, the
gripper or the robot mounting:

    py -3.12 coripps/precompute_pick_poses.py

Keys: ``space`` runs the next tube, ``q``/``esc`` quits.
"""

import os
import sys

# Allow running as `python coripps/picker_execute_picks.py` from the root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import coripps.pick_layout as layout
import coripps.pick_poses_io as ppio
import coripps.picker_motion as pkm


def load_tubes(picker, path=ppio.PICK_POSES_JSON):
    """Adapt the precomputed file to ``picker_motion``'s tube_source shape.

    The reading and the checking live in ``pick_poses_io``; all this adds is
    the ``(points, sequences)`` signature ``main`` expects, and the one-line
    report of what was loaded.
    """
    points, sequences, _rotmats, meta = ppio.load_checked(
        path, ndof=picker.ndof, pick_z=layout.PICK_Z)
    print(f'tubes: {len(points)} precomputed picks from '
          f'{os.path.basename(path)} '
          f"(grid {meta.get('grid_total')}, "
          f"grasp height {meta.get('pick_z')})")
    return points, sequences


def main():
    return pkm.main(tube_source=load_tubes)


if __name__ == '__main__':
    pkm.run(main())
