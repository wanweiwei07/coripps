"""Calibration of the sampler's two webcams (left / right, ``camera``):
their settings (folders, result files, robot) and, run, a menu of the
steps -- the work is done by ``one.vision.calibrate``.

    py -3.12 coripps/vision/sampler_stereo/calib.py

First a live view of both pictures side by side, to check / aim them.
Then the images for both lens fits and the stereo fit in one pass
(``one.vision.calibrate.stereo.capture``): both pictures live side by
side, space saves each picture that sees enough board corners (a pair
when both do), q quits. Hold the board still for the space (the two are
not synchronised); reach every part of each picture, corners and edges
most, at different tilts and distances, and 15-20 of the views where
both see it. 25-35 in all.

Then, for the left camera and again for the right:

* lens fit -- the fisheye model from its images of that pass.
* hand-eye solve again -- from the images of the hand-eye step below,
  nothing moves.

Hand-eye of both at once (``one.vision.calibrate.auto_capture``), after
both lens fits and the stereo fit: the board flat on the table, both
cameras seeing it from where the sampler stands (the picker is read once
and frozen as an obstacle, the platform as PLATFORM_BOXES). The arm moves
around it -- views planned for each camera, both record at every pose
(the plan is printed first, 'g' starts each phase, q / Esc stops) --
then the joint hand-eye below.

And for the pair (``one.vision.calibrate.stereo``), after both lens fits:

* stereo fit -- the right camera's pose in the left one (``STEREO_PATH``)
  straight from the pairs of that pass.
* joint hand-eye -- after the stereo fit, from the recorded hand-eye
  runs (nothing moves): each hand-eye solved on its own, the two put
  together checked against the stereo pose (the arm's error in one, not
  in the other), then the left camera's hand-eye from BOTH runs' images,
  the right held at the stereo pose from it -- both results agree with
  the stereo pose.
"""

import os
import sys

import numpy as np

from one.vision.calibrate import lens
from one.vision.calibrate import handeye as che

from coripps.vision.sampler_stereo.camera import PORTS, LABELS


_HERE = os.path.dirname(os.path.abspath(__file__))
# coripps/config (this file is coripps/vision/<camera>/calib.py)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'config')
_ROBOT = 2                      # mobisys(): (base, picker, sampler)
# the platform (mobisys' cart box, tube_rack / capper .stl) as boxes for the hand-eye moves: its
# mesh is too fine for the collider, and what stands on the table is two
# low blocks; (x, y, z) min / max in the world, m, ~1 cm margin
PLATFORM_BOXES = (((-0.25, 0.09, 0.0), (0.31, 0.26, 0.055)),
                  ((0.09, -0.13, 0.0), (0.31, 0.10, 0.120)))  # capper +25 mm
# hand-eye views: camera to board centre (m), near the working range
# (100-300 mm) -- but the cutter's tube reaches ~10 cm past the cameras,
# so not nearer than where its tip stays ~5 cm off the board, also at
# handeye_capture's 35 deg tilt
HANDEYE_DISTANCES = (0.16, 0.20)
# the right camera's pose in the left camera's frame (stereo.fit)
STEREO_PATH = os.path.join(CONFIG_DIR, 'sampler_stereo.json')


def platform_boxes():
    """:data:`PLATFORM_BOXES` as scene objects with collision."""
    import one.scene.scene_object_primitive as ossop
    import one.utils.constant as ouc
    boxes = []
    for lo, hi in PLATFORM_BOXES:
        lo, hi = np.asarray(lo), np.asarray(hi)
        boxes.append(ossop.box(pos=(lo + hi) / 2, xyz_lengths=hi - lo,
                               collision_type=ouc.CollisionType.AABB))
    return boxes


def charuco_dir(side):
    return os.path.join(_HERE, f'calib_images_{side}')


def handeye_dir(side):
    return os.path.join(_HERE, f'handeye_images_{side}')


def intrinsics_path(side):
    return os.path.join(CONFIG_DIR, f'sampler_{side}_cam_intrinsics.json')


def handeye_path(side):
    return os.path.join(CONFIG_DIR, f'sampler_{side}_cam_handeye.json')


def load_intrinsics(side):
    """``(K, D)`` of ``side``'s fisheye model."""
    return lens.load_intrinsics(intrinsics_path(side))


def load_handeye(side):
    """``side``'s 4x4 flange -> camera tf."""
    return che.load_handeye(handeye_path(side))


def load_stereo():
    """The right camera's 4x4 pose in the left camera's frame."""
    from one.vision.calibrate import stereo
    return stereo.load_stereo(STEREO_PATH)


def detector(side):
    return che.BoardDetector(*load_intrinsics(side), fisheye=True)


def flange_tfs(qs_list):
    """The sampler's flange tf in the world for each joint vector."""
    from coripps.robots.mobisys import mobisys
    return che.flange_tfs(mobisys()[_ROBOT], qs_list)


def fit_lens(side):
    return lens.fit(charuco_dir(side), intrinsics_path(side))


def solve(side):
    from coripps.robots.mobisys import mobisys
    return che.calibrate(mobisys()[_ROBOT], detector(side),
                         handeye_dir(side), handeye_path(side))


def capture_handeye():
    """Both cameras' hand-eye images in one run of the arm, then each
    solved and the two as one fit (:func:`solve_joint`)."""
    from one.vision.calibrate import auto_capture as cac
    from one.vision.calibrate.handeye_capture import WebCam
    from coripps.tool.home_sampler import _build, _SAMPLER_HOST
    data_dirs = {side: handeye_dir(side) for side in PORTS}

    def make_session():
        cameras = {}
        try:
            for side in PORTS:
                cameras[side] = WebCam(PORTS[side], LABELS[side])
        except RuntimeError:
            for cam in cameras.values():
                cam.close()
            raise
        return cac.Session(build=lambda: _build(obstacles=platform_boxes()),
                           host=_SAMPLER_HOST, cameras=cameras,
                           detectors={side: detector(side) for side in PORTS},
                           data_dirs=data_dirs, distances=HANDEYE_DISTANCES)

    return cac.run(make_session, data_dirs, solve_joint)


def capture_images():
    """The images of both lens fits and the stereo fit, both cameras at
    once (``stereo.capture``)."""
    from one.vision.calibrate import stereo
    return stereo.capture(PORTS['left'], PORTS['right'], charuco_dir('left'),
                          charuco_dir('right'),
                          (LABELS['left'], LABELS['right']))


def fit_stereo():
    """Left -> right straight from the image pairs (the lens fits fixed)."""
    from one.vision.calibrate import stereo
    return stereo.fit(charuco_dir('left'), charuco_dir('right'),
                      intrinsics_path('left'), intrinsics_path('right'),
                      STEREO_PATH)


def compare_handeyes():
    """Print how far the two hand-eyes (each solved on its own) put
    together are from the stereo pose; True."""
    from_handeyes = np.linalg.inv(load_handeye('left')) \
        @ load_handeye('right')
    diff = np.linalg.inv(load_stereo()) @ from_handeyes
    angle = np.degrees(np.arccos(np.clip((np.trace(diff[:3, :3]) - 1)
                                         / 2, -1, 1)))
    print(f'the two hand-eyes put together differ from the stereo pose by '
          f'{np.linalg.norm(diff[:3, 3]) * 1000:.2f} mm, {angle:.2f} deg')
    return True


def solve_joint():
    """Each hand-eye solved on its own from its recorded run, the two put
    together checked against the stereo pose (``fit_stereo``), then both
    as one fit, the right camera held at the stereo pose from the left
    (``stereo.calibrate_handeye``); overwrites both hand-eye results."""
    from one.vision.calibrate import stereo
    from coripps.robots.mobisys import mobisys
    if not all(solve(side) for side in PORTS):
        return False
    compare_handeyes()
    return stereo.calibrate_handeye(
        mobisys()[_ROBOT], detector('left'), handeye_dir('left'),
        detector('right'), handeye_dir('right'), load_stereo(),
        handeye_path('left'), handeye_path('right'))


if __name__ == '__main__':
    from one.vision.calibrate import menu
    from coripps.vision.sampler_stereo.camera import live
    from coripps.vision.sampler_stereo import depth
    steps = [('live view, left | right', live),
             ('lens + stereo images: both live, space saves', capture_images)]
    for side in PORTS:
        steps += [
            (f'{side}: lens fit', lambda s=side: fit_lens(s)),
            (f'{side}: hand-eye: solve again from the recorded images',
             lambda s=side: solve(s)),
        ]
    steps += [('stereo fit: left -> right from the pairs', fit_stereo),
              ('hand-eye, both cameras: move around the board, record, '
               'solve each and jointly', capture_handeye),
              ('joint hand-eye: each solved, checked against the stereo '
               'pose, then both as one fit', solve_joint),
              ('depth: live, rectified left | disparity', depth.live),
              ('depth: the cloud in the world with the robots (read-only)',
               depth.world)]
    sys.exit(0 if menu.run('sampler stereo calibration:', steps) else 1)
