"""The pick-pose file contract: what ``precompute_poses_pick`` writes and what
the execution scripts read.

It sits between the two on purpose. Running a pick should not mean importing
the solver -- the executors need the FILE, not the grid or the reachability
filter. Anything both sides have to agree on lives here:

* where the file is,
* how one entry is shaped: the grasp point and the joints of its
  ``pre_pick`` (``layout_pick``: the hand looking at the cap from above).
  The rest of a pick depends on the cap's long tab, so the executor solves
  it at run time; every point in the file can be picked at every tab
  direction,
* where the tube model sits relative to a pick point.

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

# Bumped whenever the entry shape changes, so a stale file is refused rather
# than misread. 3: from above by the cap -- a point and its pre_pick joints
# (2: a side approach, a joint path through four poses)
FORMAT_VERSION = 3

# Where the tube model stands relative to the TCP pick point: the grasp is at
# the cap's top, 57 mm above the tube model's origin.
TUBE_POS_OFFSET = np.array([0.0, 0.0, -0.057], dtype=np.float32)
TUBE_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 2.0)


def save(points, look_qs, path=PICK_POSES_JSON, pick_z=None, look_rise=None,
         lift_dist=None, grid_total=None):
    """Write the pickable points and their pre_pick joints to JSON."""
    picks = [{'point': np.asarray(p, dtype=np.float32).tolist(),
              'look_qs': np.asarray(q, dtype=np.float32).tolist()}
             for p, q in zip(points, look_qs)]
    data = {
        'metadata': {
            'format': FORMAT_VERSION,
            'robot': 'TubePicker',
            'tcp': 'grasp_center',
            'pick_z': None if pick_z is None else float(pick_z),
            'look_rise': None if look_rise is None else float(look_rise),
            'lift_dist': None if lift_dist is None else float(lift_dist),
            'grid_total': None if grid_total is None else int(grid_total),
            'reachable': len(picks),
        },
        'picks': picks,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'saved {len(picks)} pick points -> {path}')
    return path


def load(path=PICK_POSES_JSON):
    """``(points (n, 3), look_qs (n, ndof), metadata)``."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    picks = data['picks']
    points = np.asarray([p['point'] for p in picks], dtype=np.float32)
    look_qs = np.asarray([p['look_qs'] for p in picks], dtype=np.float32)
    return points, look_qs, data['metadata']


def load_checked(path=PICK_POSES_JSON, ndof=None, layout=None):
    """:func:`load`, refusing a file that does not match: missing, another
    format, another robot, or solved from other ``layout_pick`` numbers
    (``layout``: that module) -- it raises rather than run poses nobody
    reviewed."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing -- run tool/tube/precompute_poses_pick.py')
    with open(path, encoding='utf-8') as f:
        version = json.load(f)['metadata'].get('format', 1)
    if version != FORMAT_VERSION:
        raise ValueError(
            f'{path} is format {version}, this build reads format '
            f'{FORMAT_VERSION} -- regenerate it with '
            'tool/tube/precompute_poses_pick.py')
    points, look_qs, meta = load(path)
    if len(points) == 0:
        raise ValueError(f'{path} holds no pick point')
    if ndof is not None and look_qs.shape[-1] != ndof:
        raise ValueError(
            f'{path} holds {look_qs.shape[-1]}-joint configs, the robot has '
            f'{ndof} -- regenerate it')
    if layout is not None:
        for key, value in (('pick_z', layout.PICK_Z),
                           ('look_rise', layout.PICK_LOOK_RISE),
                           ('lift_dist', layout.LIFT_DIST)):
            stored = meta.get(key)
            if stored is None or not np.isclose(stored, value):
                raise ValueError(
                    f'{path} was solved with {key} {stored}, layout_pick has '
                    f'{value} -- regenerate it')
    return points, look_qs, meta
