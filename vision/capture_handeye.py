"""Move the sampler through ~20 poses that look at the ChArUco board on the
table, record an image and the joint angles at each, then solve the
hand-eye calibration (``calibrate_handeye``).

Start with the left camera seeing the whole board (the board lies flat on the
table, at about the robot base's height). Two phases:

1. bootstrap -- small turns of J4 / J5 / J6 alone around the start pose
   (the board stays in view); they give a rough flange -> camera transform.
2. views -- with that, camera poses on a cap over the board centre: tilted
   0-35 deg from vertical in six directions, two distances, a little roll.
   Kept only when there is an IK solution, the arm is collision free (the
   picker is read once and frozen as an obstacle) and the whole board is
   predicted to be in the picture.

Everything is planned before anything moves; the plan is printed and 'g'
starts it. q / Esc stops before the next pose. At the end the arm goes back
to the start pose. The images and joint angles go to ``handeye_images/``
(cleared at the start of each run).

    py -3.12 coripps/vision/capture_handeye.py
"""

import os
import sys
import glob
import json
import time

import cv2
import numpy as np

import one.motion.probabilistic.rrt as ompr
from one.drivers.cobotta import BCapError, CobottaMotion

from coripps.tool.home_sampler import _build, _SAMPLER_HOST
from coripps.tool.pick_sequence_single import _quit_pressed, _wait_for_g
from coripps.vision.cameras import find_by_port
from coripps.vision.capture_charuco import PORT
from coripps.vision.charuco_board import NROW, NCOLUMN, SQUARE_SIZE
from coripps.vision import calibrate_handeye as che


SPEED = 30.0
SETTLE = 0.5                    # s after arriving, before the image
FLUSH_FRAMES = 10               # stale frames the camera has buffered
# bootstrap: (joint index, deg) turns from the start pose
BOOTSTRAP = ((3, 10), (3, -10), (4, 10), (4, -10), (5, 15), (5, -15))
TILTS_DEG = (20, 35, 0)         # camera axis from vertical
AZIMUTHS_DEG = (0, 60, 120, 180, 240, 300)
DISTANCES = (0.20, 0.25)        # m, camera to board centre
ROLLS_DEG = (0, 20, -20)
N_VIEWS = 14
IMAGE_MARGIN = 15               # px, board corners kept this far inside
_EXTEND_STEP_SIZE = np.pi / 36
_MAX_ITERS = 3000
_SHORTCUT_ITERS = 200


def _rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, dtype=np.float64)
                         * np.deg2rad(deg))[0]


class Camera:
    """The left camera, kept open for the whole run."""

    def __init__(self):
        by_port = find_by_port()
        if PORT not in by_port:
            raise RuntimeError(f'no camera on hub port {PORT} '
                               f'(found: {by_port})')
        self.cap = cv2.VideoCapture(by_port[PORT], cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError('left camera: cannot open')

    def frame(self):
        """A frame taken now, not one from the buffer."""
        for _ in range(FLUSH_FRAMES):
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError('left camera: gives no image')
        return frame

    def close(self):
        self.cap.release()


class Session:
    def __init__(self):
        _, _, self.sampler, self.pln_ctx = _build()
        self.detector = che.BoardDetector()
        self.camera = Camera()
        self.cobotta = CobottaMotion(_SAMPLER_HOST, speed=SPEED).connect()
        self.records = []
        self.img_size = None

    def close(self):
        self.camera.close()
        self.cobotta.close()            # powers off, gives the arm back

    # -- moving -----------------------------------------------------------
    def qs_now(self):
        return self.pln_ctx.enforce_bounds(
            self.cobotta.joint_angles(ndof=self.sampler.ndof))

    def path_to(self, start, goal):
        """A collision-free joint path, or None."""
        if self.pln_ctx.is_motion_valid(start, goal):
            return [start, goal]
        planner = ompr.RRTConnectPlanner(pln_ctx=self.pln_ctx,
                                         extend_step_size=_EXTEND_STEP_SIZE)
        path = planner.solve(start=start, goal=goal, max_iters=_MAX_ITERS)
        if not path:
            return None
        return ompr.shortcut_path(path, self.pln_ctx, n_iter=_SHORTCUT_ITERS)

    def go(self, goal):
        """Move to ``goal``; False when no path."""
        path = self.path_to(self.qs_now(), goal)
        if path is None:
            print('  no collision-free path -- skipped')
            return False
        self.cobotta.move_path(path)
        return True

    # -- recording --------------------------------------------------------
    def record(self, label):
        """Image + measured joint angles at the current pose; the board's
        camera tf, or None when it is not seen well enough."""
        time.sleep(SETTLE)
        image = self.camera.frame()
        self.img_size = image.shape[1::-1]
        qs = self.cobotta.joint_angles(ndof=self.sampler.ndof)
        tf, n = self.detector.pose(image)
        name = f'{len(self.records):03d}.png'
        cv2.imwrite(os.path.join(che.DATA_DIR, name), image)
        self.records.append({'image': name, 'qs': [float(q) for q in qs],
                             'label': label})
        print(f'  {name} {label}: {n} corners'
              + ('' if tf is not None else ' -- too few, not usable'))
        return tf

    # -- planning ---------------------------------------------------------
    def bootstrap_goals(self, start):
        goals = []
        for jnt, deg in BOOTSTRAP:
            goal = start.copy()
            goal[jnt] += np.deg2rad(deg)
            if self.pln_ctx.is_motion_valid(start, goal):
                goals.append((f'J{jnt + 1} {deg:+d} deg', goal))
            else:
                print(f'bootstrap J{jnt + 1} {deg:+d} deg: out of limits or '
                      'in collision -- left out')
        return goals

    def plan_views(self, start, tf_flange_cam, tf_w_board):
        """Up to N_VIEWS ``(label, qs)`` around the board, nearest-first
        from ``start``."""
        center_board = np.array([NCOLUMN * SQUARE_SIZE / 2,
                                 NROW * SQUARE_SIZE / 2, 0.0, 1.0])
        center = (tf_w_board @ center_board)[:3]
        self.sampler.fk(qs=start)
        tf_w_cam0 = (np.array(self.sampler.tcp('flange').tf, dtype=np.float64)
                     @ tf_flange_cam)
        x_ref = tf_w_cam0[:3, 0]
        tf_cam_flange = np.linalg.inv(tf_flange_cam)
        views = []
        # one view per (tilt, direction): the first distance / roll that
        # works, the preferred distance and roll changing from view to view
        for tilt in TILTS_DEG:
            for azim in (AZIMUTHS_DEG if tilt else (0,)):
                if len(views) >= N_VIEWS:
                    break
                up = _rot((0, 0, 1), azim) @ _rot((0, 1, 0), tilt) \
                    @ np.array([0.0, 0.0, 1.0])
                z = -up
                x = x_ref - (x_ref @ z) * z
                x /= np.linalg.norm(x)
                k_d = len(views) % len(DISTANCES)
                k_r = len(views) % len(ROLLS_DEG)
                view = None
                for dist in DISTANCES[k_d:] + DISTANCES[:k_d]:
                    for roll in ROLLS_DEG[k_r:] + ROLLS_DEG[:k_r]:
                        rx = _rot(z, roll) @ x
                        tf_w_cam = che._tf(center + dist * up, np.column_stack(
                            [rx, np.cross(z, rx), z]))
                        qs = self._reachable(tf_w_cam @ tf_cam_flange, start)
                        if qs is not None and self._sees_board(
                                np.linalg.inv(tf_w_cam) @ tf_w_board):
                            view = (f'tilt {tilt} azim {azim} dist '
                                    f'{dist * 100:.0f}cm roll {roll:+d}', qs)
                            break
                    if view is not None:
                        break
                if view is None:
                    print(f'view tilt {tilt} azim {azim}: no reachable, '
                          'collision-free pose that sees the board')
                else:
                    views.append(view)
        # visit nearest-first so the arm does not cross back and forth
        ordered, cur = [], start
        while views:
            i = int(np.argmin([np.max(np.abs(q - cur)) for _, q in views]))
            ordered.append(views.pop(i))
            cur = ordered[-1][1]
        return ordered

    def _reachable(self, tf_w_flange, near):
        sols = self.sampler.ik(tf_w_flange[:3, 3].astype(np.float32),
                               tf_w_flange[:3, :3].astype(np.float32),
                               ref_qs=near)
        best = None
        for qs in sols:
            qs = np.asarray(qs, dtype=np.float32)
            if not self.pln_ctx.is_state_valid(qs):
                continue
            if best is None or (np.max(np.abs(qs - near))
                                < np.max(np.abs(best - near))):
                best = qs
        return best

    def _sees_board(self, tf_cam_board):
        px = self.detector.project(tf_cam_board)
        w, h = self.img_size
        m = IMAGE_MARGIN
        return bool(np.all((px[:, 0] > m) & (px[:, 0] < w - m)
                           & (px[:, 1] > m) & (px[:, 1] < h - m)))

    # -- the run ----------------------------------------------------------
    def poses_seen(self):
        """``(flange tfs, detected board corners)`` of the usable
        records."""
        qs_list, observations = [], []
        for rec in self.records:
            image = cv2.imread(os.path.join(che.DATA_DIR, rec['image']))
            found, _ = self.detector.corners(image)
            if found is not None:
                qs_list.append(rec['qs'])
                observations.append(found)
        return che.flange_tfs(qs_list), observations

    def save(self):
        with open(che.RECORD_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)

    def run(self):
        start = self.qs_now()
        print(f'start: {np.round(np.degrees(start), 2)} deg')
        if not self.pln_ctx.is_state_valid(start):
            print('start: in collision in the model -- refusing to move')
            return False
        if self.record('start') is None:
            print('the board is not seen well from the start pose -- point '
                  'the camera at it first')
            return False
        boot = self.bootstrap_goals(start)
        print(f'\nphase 1: {len(boot)} bootstrap turns around the start pose '
              f'at {SPEED:.0f}%:')
        for label, _ in boot:
            print(f'  {label}')
        if not _wait_for_g('bootstrap'):
            return False
        self.cobotta.enable()
        try:
            for label, goal in boot:
                if _quit_pressed():
                    print('stopped')
                    return False
                print(label)
                if self.go(goal):
                    self.record(label)
            if not self.go(start):
                print('cannot get back to the start pose')
                return False
            flanges, observations = self.poses_seen()
            if len(observations) < 4:
                print(f'only {len(observations)} usable bootstrap images -- '
                      'stopping')
                return False
            print('\nrough hand-eye from the bootstrap:')
            tf_flange_cam, tf_w_board, per_image = che.solve(
                self.detector, flanges, observations)
            che.report(tf_flange_cam, tf_w_board, per_image)
            views = self.plan_views(start, tf_flange_cam, tf_w_board)
            print(f'\nphase 2: {len(views)} views:')
            for label, qs in views:
                print(f'  {label}: {np.round(np.degrees(qs), 1)} deg')
            if not views:
                return False
            self.save()
            if not _wait_for_g('views'):
                return False
            for label, goal in views:
                if _quit_pressed():
                    print('stopped')
                    break
                print(label)
                if self.go(goal):
                    self.record(label)
                    self.save()
            print('back to the start pose')
            self.go(start)
        finally:
            self.save()
        return True


def main():
    os.makedirs(che.DATA_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(che.DATA_DIR, '*.png')):
        os.remove(path)
    try:
        session = Session()
    except (OSError, BCapError, RuntimeError) as exc:
        print(exc)
        return False
    try:
        if not session.run():
            return False
    except (OSError, BCapError) as exc:
        print(f'sampler: {exc} -- stopping')
        return False
    finally:
        session.close()
    print('\nfinal hand-eye calibration:')
    return che.main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
