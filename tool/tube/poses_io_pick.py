"""The pick-pose file contract: what ``precompute_poses_pick`` writes and what
the execution scripts read.

It sits between the two on purpose. Running a pick should not mean importing
the solver -- the executors need the FILE, not the IK, the grid or the
reachability filter. Anything both sides have to agree on lives here:

* where the file is and what the four poses are called,
* how one entry is shaped (point, tcp orientation, the joint PATH through the
  four poses, sampled along straight TCP lines),
* where the tube model sits relative to a pick point, which is what turns a
  TCP position back into "a tube standing there".

Nothing in this module solves or plans, so it stays cheap to import.
"""

import os
import json

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc


# coripps/config, next to the package's other data -- not tool/config
PICK_POSES_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'config', 'poses_pick.json')

# The four poses of one pick, in execution order. 'pick' is the point itself;
# the other three are offsets along the TCP's own axes.
POSE_NAMES = ('pre_pick_high', 'pre_pick', 'pick', 'lift')

# Bumped whenever the entry shape changes, so a stale file is refused rather
# than misread. 2: a dense straight-line joint PATH per pick instead of the
# four bare poses (the joint-space moves between those bowed sideways).
FORMAT_VERSION = 2

# Where the tube model stands relative to the TCP pick point: the grasp is
# taken this far up the tube, and the tube is turned to face the gripper.
TUBE_POS_OFFSET = np.array([0.0, 0.0, -0.045], dtype=np.float32)
TUBE_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 2.0)


def save(points, paths, marks, rotmats, path=PICK_POSES_JSON,
         pick_z=None, grid_total=None, path_step=None):
    """Write the reachable pick points and their joint paths to JSON.

    One entry per point: the TCP position, the TCP orientation it solved with
    (the grid mixes the spun and unspun orientations, so replay needs it) and
    the joint PATH -- every straight-line sample from ``pre_pick_high`` to
    ``lift``. ``marks[k]`` is the path index of key pose ``POSE_NAMES[k]``;
    every pick shares it, so it lives in the metadata.
    """
    picks = []
    for point, joint_path, rotmat in zip(points, paths, rotmats):
        picks.append({
            'point': np.asarray(point, dtype=np.float32).tolist(),
            'tcp_rotmat': np.asarray(rotmat, dtype=np.float32).tolist(),
            'path': np.asarray(joint_path, dtype=np.float32).tolist(),
        })
    data = {
        'metadata': {
            'format': FORMAT_VERSION,
            'robot': 'TubePicker',
            'tcp': 'grasp_center',
            'pose_names': list(POSE_NAMES),
            'marks': [int(m) for m in marks],
            'path_step': None if path_step is None else float(path_step),
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
            f'{path} is missing -- run tool/precompute_poses_pick.py first')
    with open(path, encoding='utf-8') as f:
        version = json.load(f)['metadata'].get('format', 1)
    if version != FORMAT_VERSION:
        raise ValueError(
            f'{path} is format {version}, this build reads format '
            f'{FORMAT_VERSION} -- regenerate it with '
            'tool/precompute_poses_pick.py')
    points, paths, rotmats, meta = load(path)
    if len(points) == 0:
        raise ValueError(f'{path} holds no reachable pick point')
    names = tuple(meta.get('pose_names', ()))
    if names != POSE_NAMES:
        raise ValueError(
            f'{path} was written with poses {names}, this build expects '
            f'{POSE_NAMES} -- regenerate it')
    if ndof is not None and paths.shape[-1] != ndof:
        raise ValueError(
            f'{path} holds {paths.shape[-1]}-joint configs, the robot has '
            f'{ndof} -- regenerate it')
    stored_z = meta.get('pick_z')
    if (pick_z is not None and stored_z is not None
            and not np.isclose(stored_z, pick_z)):
        raise ValueError(
            f'{path} was solved at grasp height {stored_z}, this run uses '
            f'{pick_z} -- regenerate it or align the layout')
    return points, paths, rotmats, meta


def load(path=PICK_POSES_JSON):
    """Read back what :func:`save` wrote as
    ``(points, paths, rotmats, metadata)``; ``paths`` is
    ``(n, n_samples, ndof)`` and ``metadata['marks']`` indexes the key poses
    in it (see :func:`key_poses`)."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    picks = data['picks']
    points = np.asarray([p['point'] for p in picks], dtype=np.float32)
    rotmats = np.asarray([p['tcp_rotmat'] for p in picks], dtype=np.float32)
    paths = np.asarray([p['path'] for p in picks], dtype=np.float32)
    return points, paths, rotmats, data['metadata']


def key_poses(paths, metadata):
    """The four key poses of every path, ``(n, 4, ndof)``, in
    :data:`POSE_NAMES` order."""
    return np.asarray(paths)[:, list(metadata['marks'])]
