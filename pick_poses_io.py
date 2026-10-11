"""The pick-pose file contract: what ``precompute_pick_poses`` writes and what
the execution scripts read.

It sits between the two on purpose. Running a pick should not mean importing
the solver -- the executors need the FILE, not the IK, the grid or the
reachability filter. Anything both sides have to agree on lives here:

* where the file is and what the four poses are called,
* how one entry is shaped (point, tcp orientation, the four joint configs),
* where the tube model sits relative to a pick point, which is what turns a
  TCP position back into "a tube standing there".

Nothing in this module solves or plans, so it stays cheap to import.
"""

import os
import json

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc


_CORIPPS_DIR = os.path.dirname(os.path.abspath(__file__))
PICK_POSES_JSON = os.path.join(_CORIPPS_DIR, 'config', 'pick_poses.json')

# The four poses of one pick, in execution order. 'pick' is the point itself;
# the other three are offsets along the TCP's own axes.
POSE_NAMES = ('pre_pick_high', 'pre_pick', 'pick', 'lift')

# Where the tube model stands relative to the TCP pick point: the grasp is
# taken this far up the tube, and the tube is turned to face the gripper.
TUBE_POS_OFFSET = np.array([0.0, 0.0, -0.045], dtype=np.float32)
TUBE_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 2.0)


def save(points, sequences, rotmats, path=PICK_POSES_JSON,
         pick_z=None, grid_total=None):
    """Write the reachable pick points and their four pose configs to JSON.

    One entry per point: the TCP position, the TCP orientation it solved with
    (the grid mixes the spun and unspun orientations, so replay needs it) and
    the joint angles of each pose, keyed by :data:`POSE_NAMES`.
    """
    picks = []
    for point, sequence, rotmat in zip(points, sequences, rotmats):
        picks.append({
            'point': np.asarray(point, dtype=np.float32).tolist(),
            'tcp_rotmat': np.asarray(rotmat, dtype=np.float32).tolist(),
            'qs': {name: np.asarray(qs, dtype=np.float32).tolist()
                   for name, qs in zip(POSE_NAMES, sequence)},
        })
    data = {
        'metadata': {
            'robot': 'TubePicker',
            'tcp': 'grasp_center',
            'pose_names': list(POSE_NAMES),
            'pick_z': None if pick_z is None else float(pick_z),
            'grid_total': None if grid_total is None else int(grid_total),
            'reachable': len(picks),
        },
        'picks': picks,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'saved {len(picks)} pick poses -> {path}')
    return path


def load_checked(path=PICK_POSES_JSON, ndof=None, pick_z=None):
    """:func:`load`, but refusing a file that does not match expectations.

    Every check compares against a plain VALUE the caller passes in -- this
    module stays free of robots and of the scripts that consume it. It raises
    rather than falling back to solving: a mismatch means the file describes a
    different robot or a different grid, and quietly re-solving would hide
    that the arm is about to run poses nobody reviewed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing -- run tool/precompute_pick_poses.py first')
    points, sequences, rotmats, meta = load(path)
    if len(points) == 0:
        raise ValueError(f'{path} holds no reachable pick point')
    names = tuple(meta.get('pose_names', ()))
    if names != POSE_NAMES:
        raise ValueError(
            f'{path} was written with poses {names}, this build expects '
            f'{POSE_NAMES} -- regenerate it')
    if ndof is not None and sequences.shape[-1] != ndof:
        raise ValueError(
            f'{path} holds {sequences.shape[-1]}-joint configs, the robot has '
            f'{ndof} -- regenerate it')
    stored_z = meta.get('pick_z')
    if (pick_z is not None and stored_z is not None
            and not np.isclose(stored_z, pick_z)):
        raise ValueError(
            f'{path} was solved at grasp height {stored_z}, this run uses '
            f'{pick_z} -- regenerate it or align the layout')
    return points, sequences, rotmats, meta


def load(path=PICK_POSES_JSON):
    """Read back what :func:`save` wrote as
    ``(points, sequences, rotmats, metadata)`` arrays."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    picks = data['picks']
    pose_names = data['metadata']['pose_names']
    points = np.asarray([p['point'] for p in picks], dtype=np.float32)
    rotmats = np.asarray([p['tcp_rotmat'] for p in picks], dtype=np.float32)
    sequences = np.asarray(
        [[p['qs'][name] for name in pose_names] for p in picks],
        dtype=np.float32)
    return points, sequences, rotmats, data['metadata']
