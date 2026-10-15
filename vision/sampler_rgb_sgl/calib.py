"""Calibration of the left cutter-hand webcam (on the
sampler): its settings (port, folders, result
files, robot) and, run, a menu of the steps -- the work is done by
``one.vision.calibrate``.

    py -3.12 coripps/vision/sampler_rgb_sgl/calib.py

1. lens images -- live view, space saves (enough board corners only), q
   quits. Move the board (or the camera) so it reaches every part of the
   picture, corners and edges most, at different tilts and distances.
   20-30 images.
2. lens fit -- the fisheye model from those images.
3. hand-eye -- the board flat on the table, the camera seeing all of it
   from where the sampler stands (the picker is read once and frozen as
   an obstacle). The arm moves around
   it (the plan is printed first, 'g' starts each phase, q / Esc stops),
   then it is solved.
4. hand-eye solve again -- from the images of 3, nothing moves.
"""

import os
import sys

from one.vision.calibrate import lens
from one.vision.calibrate import handeye as che


PORT = 1                       # hub port of the left camera
LABEL = 'left camera'
_HERE = os.path.dirname(os.path.abspath(__file__))
# coripps/config (this file is coripps/vision/<camera>/calib.py)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'config')
CHARUCO_DIR = os.path.join(_HERE, 'calib_images')
HANDEYE_DIR = os.path.join(_HERE, 'handeye_images')
INTRINSICS_PATH = os.path.join(CONFIG_DIR, 'left_cam_intrinsics.json')
HANDEYE_PATH = os.path.join(CONFIG_DIR, 'left_cam_handeye.json')
_ROBOT = 2                      # mobisys(): (base, picker, sampler)


def load_intrinsics():
    """``(K, D)`` of the left camera's fisheye model."""
    return lens.load_intrinsics(INTRINSICS_PATH)


def load_handeye():
    """The left camera's 4x4 flange -> camera tf."""
    return che.load_handeye(HANDEYE_PATH)


def detector():
    return che.BoardDetector(*load_intrinsics(), fisheye=True)


def flange_tfs(qs_list):
    """The sampler's flange tf in the world for each joint vector."""
    from coripps.robots.mobisys import mobisys
    return che.flange_tfs(mobisys()[_ROBOT], qs_list)


def session_args(data_dir=HANDEYE_DIR):
    """The arguments of a ``handeye_capture.Session`` on the sampler with
    this camera, recording into ``data_dir``."""
    from coripps.tool.home_sampler import _build, _SAMPLER_HOST
    from one.vision.calibrate.handeye_capture import WebCam
    return dict(build=_build, host=_SAMPLER_HOST,
                camera=WebCam(PORT, LABEL), detector=detector(),
                data_dir=data_dir)


def capture_lens():
    from one.vision.calibrate import charuco
    return charuco.capture(PORT, CHARUCO_DIR, LABEL)


def fit_lens():
    return lens.fit(CHARUCO_DIR, INTRINSICS_PATH)


def solve():
    from coripps.robots.mobisys import mobisys
    return che.calibrate(mobisys()[_ROBOT], detector(), HANDEYE_DIR,
                         HANDEYE_PATH)


def capture_handeye():
    from one.vision.calibrate import handeye_capture as chc
    return chc.run(lambda: chc.Session(**session_args()), HANDEYE_DIR,
                   solve)


if __name__ == '__main__':
    from one.vision.calibrate import menu
    sys.exit(0 if menu.run(f'{LABEL} calibration:', [
        ('lens images', capture_lens),
        ('lens fit', fit_lens),
        ('hand-eye: move around the board, record, solve', capture_handeye),
        ('hand-eye: solve again from the recorded images', solve),
    ]) else 1)
