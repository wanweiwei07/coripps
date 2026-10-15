"""Hand-eye calibration of the cutter hand's D405 (on the sampler): its
settings (folders, result file, robot) and, run, a menu of the steps -- the
work is done by ``one.vision.calibrate``.

    py -3.12 coripps/vision/d405/calib.py

No lens calibration of our own: the board is found in the infrared image,
which IS the rectified depth frame -- a plain pinhole, K from the camera
(``camera.load_intrinsics``), no distortion.

1. hand-eye -- the board flat on the table, the D405 seeing all of it from
   where the sampler stands (the picker is read once and frozen as an
   obstacle). The arm moves around it (the plan is printed first, 'g'
   starts each phase, q / Esc stops), then it is solved.
2. hand-eye solve again -- from the images of 1, nothing moves.
"""

import os
import sys

import numpy as np

from one.vision.calibrate import handeye as che
from coripps.vision.d405.camera import load_intrinsics


LABEL = 'cutter-hand D405'
_HERE = os.path.dirname(os.path.abspath(__file__))
# coripps/config (this file is coripps/vision/<camera>/calib.py)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'config')
HANDEYE_DIR = os.path.join(_HERE, 'handeye_images')
HANDEYE_PATH = os.path.join(CONFIG_DIR, 'd405_handeye.json')


def detector(K=None):
    """The board in the infrared image: pinhole, no distortion."""
    K = load_intrinsics()[0] if K is None else K
    return che.BoardDetector(K, np.zeros(5), fisheye=False)


def session():
    """A ``handeye_capture.Session`` on the sampler with the D405 (its K
    saved first, for solving without the camera)."""
    from coripps.tool.home_sampler import _build, _SAMPLER_HOST
    from one.vision.calibrate.handeye_capture import Session
    from coripps.vision.d405.camera import D405
    cam = D405()
    cam.save_intrinsics()
    return Session(_build, _SAMPLER_HOST, cam, detector(cam.K), HANDEYE_DIR)


def solve():
    from coripps.robots.mobisys import mobisys
    return che.calibrate(mobisys()[2], detector(), HANDEYE_DIR, HANDEYE_PATH)


def capture_handeye():
    from one.vision.calibrate import handeye_capture as chc
    return chc.run(session, HANDEYE_DIR, solve)


if __name__ == '__main__':
    from one.vision.calibrate import menu
    sys.exit(0 if menu.run(f'{LABEL} calibration:', [
        ('hand-eye: move around the board, record, solve', capture_handeye),
        ('hand-eye: solve again from the recorded images', solve),
    ]) else 1)
