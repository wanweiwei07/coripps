"""Which way the cap's long tab points on a tube placed in the capper, seen
by the picker's webcam.

The same detection as in the rack (``inrack_cap_orientation``: the metric
top view, SAM 3's point prompt, the cap's template), only the tube stands
elsewhere: at the drop point of ``layout_cap`` (TUBE_XY), its cap top at
Z_CAP. Any pose that sees the cap works (e.g. right after letting go of
it, the hand straight above): the camera's place comes from the joints read
off the arm.

Run, a menu:

1. look -- the joints read off the picker (nothing moves), a picture, the
   cap; saved as a sample ``samples/capper/capper_<k>``.
2. saved samples -- ``samples/capper/capper_*``; any key: next, q / Esc /
   closing the window: stop.

    py -3.12 coripps/vision/picker_rgb/incapper_cap_orientation.py
"""

import os
import sys
import glob
import json

import cv2
import numpy as np

import coripps.tool.tube.layout_cap as cap
import coripps.vision.picker_rgb.inrack_cap_orientation as cor
from coripps.vision.picker_rgb import calib


TUBE_XY = cap.DROP_POS[:2]      # m (world), the tube in the capper
# m (world), cap top. Where the template fits best (IoU 0.93 at 80 mm, 0.89
# at 76 / 82): the tube settles ~7 mm below the height it was let go at
# (grasp 75 mm, the cap top 11.8 mm above the grasp)
Z_CAP = 0.080
PREFIX = 'capper_'
SAMPLE_DIR = os.path.join(cor.SAMPLE_DIR, 'capper')  # apart from the rack's


def detect(img, qs, picker=None, seg=None, tube_xy=TUBE_XY):
    """The cap in the capper in ``img``, taken with the picker at joints
    ``qs``: :func:`inrack_cap_orientation.detect`'s dict. ``tube_xy``: the
    tube's axis (world, m) -- the drop point of its pick orientation."""
    if picker is None:
        from coripps.robots.mobisys import mobisys
        picker = mobisys()[calib._ROBOT]
    return cor.detect(img, cor.camera_tf(picker, qs), tube_xy, seg=seg,
                      z=Z_CAP)


def _next_name():
    k = 0
    while os.path.exists(os.path.join(SAMPLE_DIR, f'{PREFIX}{k:03d}.png')):
        k += 1
    return f'{PREFIX}{k:03d}'


def run_look():
    """Read the joints, take a picture, find the cap; saved as a sample."""
    import coripps.picker_motion as pkm
    from coripps.robots.mobisys import mobisys
    picker = mobisys()[calib._ROBOT]
    cor.template()                      # none: say so before anything else
    seg = cor.segmenter()
    reader = pkm.connect(pkm._PICKER_HOST, 'picker', motion=False)
    if reader is None:
        return False
    try:
        qs = reader.joint_angles(ndof=picker.ndof)
    finally:
        reader.disconnect()
    picker.fk(qs=qs)
    here = np.array(picker.gripper.tcp('grasp_center').tf[:3, 3])
    print(f'grasp_center {np.round(here * 1000, 1)} mm')
    img = cor.grab()
    res = detect(img, qs, picker, seg)
    name = _next_name()
    cor.save_sample(img, qs, TUBE_XY, name, SAMPLE_DIR)
    print(f'{name} | {cor.report(res)}')
    cv2.imshow(cor.WINDOW, cor.draw(res))
    cor._close_window()
    return res['long_deg'] is not None


def run_samples():
    from coripps.robots.mobisys import mobisys
    picker = mobisys()[calib._ROBOT]
    paths = sorted(p for p in glob.glob(os.path.join(SAMPLE_DIR,
                                                     PREFIX + '*.png'))
                   if os.path.exists(p[:-4] + '.json'))
    if not paths:
        print(f'no {PREFIX}* samples in {SAMPLE_DIR}')
        return False
    seg = cor.segmenter()
    for path in paths:
        with open(path[:-4] + '.json', encoding='utf-8') as f:
            meta = json.load(f)
        res = detect(cv2.imread(path), meta['qs'], picker, seg)
        print(os.path.basename(path), '|', cor.report(res))
        cv2.imshow(cor.WINDOW, cor.draw(res))
        if cor._wait_key() in (ord('q'), 27):   # any other key: next
            break
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    from one.vision.calibrate import menu
    sys.exit(0 if menu.run('cap orientation in the capper:', [
        ('look: from where the picker stands', run_look),
        ('saved samples', run_samples),
    ]) else 1)
