"""Look at the plant from five fixed poses with the left camera and record an
image and the joint angles at each -- the input of
``reconstruct_survey.py`` (the first image, from straight above, is its
reference).

The views are fixed in the sampler's base frame: the camera at each of
VIEW_POSITIONS -- straight above VIEW_CENTER (where the plant goes), then
VIEW_STEP to the four sides, all at VIEW_HEIGHT -- looking at VIEW_CENTER. Only the roll
about the line of sight is free: of ROLL_STEP_DEG steps, the one whose
collision-free IK (the picker is read once and frozen as an obstacle) is
nearest the previous pose.

Everything is planned before anything moves; the plan is printed and 'g'
starts it. q / Esc stops before the next pose. At the end the arm stays at
the last view (not back to the start pose, which may be close over the
plant). Images and joint angles go to ``survey_images/``
(cleared at the start of each run).

    py -3.12 coripps/vision/sampler_rgb_sgl/capture_survey.py
"""

import os
import sys
import glob
import json
import time

import cv2
import numpy as np

from one.drivers.cobotta import BCapError

from coripps.tool.keys import _quit_pressed, _wait_for_g
from one.vision.calibrate import handeye as che
from one.vision.calibrate.handeye_capture import SETTLE, Session, _rot
from coripps.vision.sampler_rgb_sgl import calib


_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, 'survey_images')
RECORD_PATH = os.path.join(DATA_DIR, 'poses.json')
TABLE_Z = 0.0007                # m (world), the board's height in hand-eye
# in the sampler's base frame (m)
VIEW_CENTER = (0.240, -0.060, 0.050)
# straight above the centre first (the reference image), then small steps
# to the four sides -- only enough parallax for the depth
VIEW_HEIGHT = 0.180
VIEW_STEP = 0.025
VIEW_POSITIONS = tuple(
    (VIEW_CENTER[0] + dx, VIEW_CENTER[1] + dy, VIEW_HEIGHT)
    for dx, dy in ((0, 0), (VIEW_STEP, -VIEW_STEP), (-VIEW_STEP, -VIEW_STEP),
                   (VIEW_STEP, VIEW_STEP), (-VIEW_STEP, VIEW_STEP)))
ROLL_STEP_DEG = 15
# leaf green in HSV (OpenCV: H 0-180)
GREEN_LO = (35, 80, 60)
GREEN_HI = (90, 255, 255)


def load_handeye():
    return calib.load_handeye()


def green_mask(image):
    """Leaf pixels, specks removed."""
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
                       GREEN_LO, GREEN_HI)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def plant_leaves(image):
    """The leaf pixels of the biggest cluster of green (the leaves of one
    plant merge when grown a little) as a bool mask, or None."""
    mask = green_mask(image)
    grown = cv2.dilate(mask, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grown)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaves = (labels == biggest) & (mask > 0)
    return leaves if leaves.any() else None


class Survey(Session):
    """The hand-eye session's arm, camera and planner, recording into
    ``survey_images/`` instead."""

    def __init__(self):
        super().__init__(**calib.session_args(DATA_DIR))
        self.tf_flange_cam = load_handeye()

    def record(self, label):
        time.sleep(SETTLE)
        image = self.camera.frame()
        self.img_size = image.shape[1::-1]
        qs = self.cobotta.joint_angles(ndof=self.robot.ndof)
        name = f'{len(self.records):03d}.png'
        cv2.imwrite(os.path.join(DATA_DIR, name), image)
        self.records.append({'image': name, 'qs': [float(q) for q in qs],
                             'label': label})
        print(f'  {name} {label}')
        return image

    def save(self):
        with open(RECORD_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)

    def plan_survey(self, start):
        """``(label, qs)`` per view of VIEW_POSITIONS, in that order; None
        when one has no collision-free IK at any roll."""
        self.robot.fk(qs=start)
        tf_w_base = np.array(self.robot.tf, dtype=np.float64)
        center = (tf_w_base @ np.r_[VIEW_CENTER, 1.0])[:3]
        tf_cam_flange = np.linalg.inv(self.tf_flange_cam)
        views, prev = [], start
        for pos_base in VIEW_POSITIONS:
            pos = (tf_w_base @ np.r_[pos_base, 1.0])[:3]
            z = (center - pos) / np.linalg.norm(center - pos)
            x0 = np.cross(z, (0.0, 0.0, 1.0))
            if np.linalg.norm(x0) < 1e-6:       # looking straight down
                x0 = np.cross(z, (1.0, 0.0, 0.0))
            x0 /= np.linalg.norm(x0)
            best = None
            for roll in range(0, 360, ROLL_STEP_DEG):
                x = _rot(z, roll) @ x0
                tf_w_cam = che._tf(pos, np.column_stack([x, np.cross(z, x),
                                                         z]))
                qs = self._reachable(tf_w_cam @ tf_cam_flange, prev)
                if qs is not None and (best is None or np.max(np.abs(
                        qs - prev)) < np.max(np.abs(best[1] - prev))):
                    best = (roll, qs)
            label = f'camera at {np.round(np.array(pos_base) * 1000)} mm'
            if best is None:
                print(f'{label}: no collision-free IK at any roll')
                return None
            views.append((f'{label}, roll {best[0]}', best[1]))
            prev = best[1]
        return views

    def run(self):
        start = self.qs_now()
        print(f'start: {np.round(np.degrees(start), 2)} deg')
        if not self.pln_ctx.is_state_valid(start):
            print('start: in collision in the model -- refusing to move')
            return False
        views = self.plan_survey(start)
        if views is None:
            return False
        print(f'\nlooking at {np.round(np.array(VIEW_CENTER) * 1000)} mm '
              f'(sampler base frame), {len(views)} views at '
              f'{self.cobotta.speed:.0f}%:')
        for label, qs in views:
            print(f'  {label}: {np.round(np.degrees(qs), 1)} deg')
        if not _wait_for_g('survey'):
            return False
        self.cobotta.enable()
        try:
            for label, goal in views:
                if _quit_pressed():
                    print('stopped')
                    break
                print(label)
                if self.go(goal):
                    self.record(label)
                    self.save()
            # the arm stays at the last view: well clear of the plant, and
            # where the next step plans from
        finally:
            self.save()
        return True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(DATA_DIR, '*.png')):
        os.remove(path)
    if os.path.exists(RECORD_PATH):
        os.remove(RECORD_PATH)
    try:
        survey = Survey()
    except (OSError, BCapError, RuntimeError) as exc:
        print(exc)
        return False
    try:
        return survey.run()
    except (OSError, BCapError) as exc:
        print(f'sampler: {exc} -- stopping')
        return False
    finally:
        survey.close()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
