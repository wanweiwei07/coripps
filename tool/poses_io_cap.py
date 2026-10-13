"""The cap-pose file contract: what ``precompute_poses_cap`` writes and what
``pick_cap_store`` reads.

Like ``poses_io_pick`` for the pick: the executor needs the FILE, not the IK.
Nothing here solves or plans, so it stays cheap to import.

The file holds, in radians:

* ``orientations[name]`` -- the PLACING, per pick orientation
  (``layout_cap.ORIENTATIONS``): ``drop`` (one config), ``place`` (the
  straight placing path, ``drop`` first) and ``release`` (the short rise
  after letting go, the placed config first);
* ``fixed`` -- the same for every tube: ``push_top`` (one config), ``push``
  (path, ``push_top`` first), ``pop`` (one config), ``press`` (path, ``pop``
  first), ``repick_top`` (one config), ``repick`` (path, ``repick_top``
  first) and ``discard`` (one config);
* ``metadata['inputs']`` -- every ``layout_cap`` number it was solved from.
"""

import os
import json

import numpy as np


# coripps/config, next to the package's other data -- not tool/config
CAP_POSES_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'poses_cap.json')
# bumped whenever the shape below changes, so a stale file is refused
FORMAT_VERSION = 3              # 3: everything after the placing is 'fixed'

_ORIENTATION_KEYS = ('drop', 'place', 'release')
_FIXED_KEYS = ('push_top', 'push', 'pop', 'press', 'repick_top', 'repick',
               'discard')


def _lists(value):
    return np.asarray(value, dtype=np.float32).tolist()


def save(orientations, fixed, inputs, path=CAP_POSES_JSON):
    """Write the solved cap poses; see the module docstring for the shape."""
    data = {
        'metadata': {'format': FORMAT_VERSION, 'robot': 'TubePicker',
                     'tcp': 'grasp_center', 'inputs': inputs},
        'orientations': {name: {key: _lists(entry[key])
                                for key in _ORIENTATION_KEYS}
                         for name, entry in orientations.items()},
        'fixed': {key: _lists(fixed[key]) for key in _FIXED_KEYS},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'saved cap poses -> {path}')
    return path


def load(path=CAP_POSES_JSON):
    """``(orientations, fixed, metadata)`` with float32 arrays."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    orientations = {name: {key: np.asarray(entry[key], dtype=np.float32)
                           for key in _ORIENTATION_KEYS}
                    for name, entry in data['orientations'].items()}
    fixed = {key: np.asarray(data['fixed'][key], dtype=np.float32)
             for key in _FIXED_KEYS}
    return orientations, fixed, data['metadata']


def load_checked(path=CAP_POSES_JSON, ndof=None, inputs=None):
    """:func:`load`, refusing a missing, old-format or stale file.

    ``inputs`` is ``layout_cap.inputs()``: when the layout has been edited
    since the file was solved, the file is refused -- re-run
    ``precompute_poses_cap.py`` -- rather than run with the old numbers.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing -- run tool/precompute_poses_cap.py first')
    with open(path, encoding='utf-8') as f:       # format first: an old
        version = json.load(f)['metadata'].get('format')   # shape won't load
    if version != FORMAT_VERSION:
        raise ValueError(f'{path} is format {version}, this build reads '
                         f'{FORMAT_VERSION} -- re-run '
                         'tool/precompute_poses_cap.py')
    orientations, fixed, meta = load(path)
    if ndof is not None and fixed['pop'].shape[-1] != ndof:
        raise ValueError(f'{path} holds {fixed["pop"].shape[-1]}-joint '
                         f'configs, the robot has {ndof} -- regenerate it')
    if inputs is not None:
        stored = meta.get('inputs', {})
        changed = [key for key, value in inputs.items()
                   if key not in stored
                   or not np.allclose(np.asarray(stored[key], dtype=float),
                                      np.asarray(value, dtype=float))]
        if changed:
            raise ValueError(f'{path} was solved from different layout_cap '
                             f'values ({", ".join(changed)}) -- re-run '
                             'tool/precompute_poses_cap.py')
    return orientations, fixed, meta
