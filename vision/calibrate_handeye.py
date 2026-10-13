"""Hand-eye calibration of the left cutter-hand webcam: where the camera sits
on the sampler's flange.

Input: the images and joint angles recorded by ``capture_handeye.py`` (the
board lies still on the table, the robot looks at it from many poses).
Unknowns: the flange -> camera tf X and the board's pose in the world W.
For image i with flange pose F_i (FK of the measured joints), the board
corners must project to where they were detected:

    corner pixel  =  fisheye( (F_i X)^-1 W  * corner on the board )

X and W (12 numbers) are fitted to all corners of all images at once by
nonlinear least squares on the pixel error (Huber loss, so a few bad
corners do not pull). The start value comes from OpenCV's closed-form
hand-eye (Park), which is only an initial guess here.

Prints the pixel residual (overall and per image) and the board's height
and tilt as a check -- it lies flat on the table at about the robot base's
height, which the fit does not know. The result goes to
``coripps/config/left_cam_handeye.json``.

    py -3.12 coripps/vision/calibrate_handeye.py
"""

import os
import sys
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

import one.utils.math as oum

from coripps.robots.mobisys import mobisys
from coripps.vision.charuco_board import charuco_board
from coripps.vision.capture_charuco import MIN_CORNERS
from coripps.vision.calibrate_intrinsics import OUT_PATH as INTRINSICS_PATH


_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, 'handeye_images')
RECORD_PATH = os.path.join(DATA_DIR, 'poses.json')
OUT_PATH = os.path.join(os.path.dirname(_HERE), 'config',
                        'left_cam_handeye.json')
HUBER_PX = 1.0                  # residuals beyond this count linearly
N_WORST = 5


def _tf(pos, rotmat):
    """4x4 tf in float64 (``oum.tf_from_pos_rotmat`` gives float32)."""
    tf = np.eye(4)
    tf[:3, :3] = rotmat
    tf[:3, 3] = np.ravel(pos)
    return tf


def _tf_from_vec(v):
    """4x4 tf from (rotation vector, position)."""
    return _tf(v[3:], cv2.Rodrigues(np.ascontiguousarray(v[:3]))[0])


def _vec_from_tf(tf):
    return np.concatenate([cv2.Rodrigues(np.ascontiguousarray(
        tf[:3, :3]))[0].ravel(), tf[:3, 3]])


def load_intrinsics():
    """``(K, D)`` of the left camera's fisheye model."""
    with open(INTRINSICS_PATH, encoding='utf-8') as f:
        cal = json.load(f)
    return np.array(cal['K']), np.array(cal['D'])


class BoardDetector:
    """The board's corners and pose in the left camera from one image."""

    def __init__(self):
        self.board = charuco_board()
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.K, self.D = load_intrinsics()

    def corners(self, image):
        """``(board points (n, 3), pixels (n, 2))`` of the detected corners,
        or None below MIN_CORNERS; and n."""
        corners, ids, _, _ = self.detector.detectBoard(image)
        n = 0 if ids is None else len(ids)
        if n < MIN_CORNERS:
            return None, n
        obj, img = self.board.matchImagePoints(corners, ids)
        return (obj.reshape(-1, 3).astype(np.float64),
                img.reshape(-1, 2).astype(np.float64)), n

    def pose(self, image):
        """``(4x4 board-in-camera tf, n corners)``; tf None below
        MIN_CORNERS."""
        found, n = self.corners(image)
        if found is None:
            return None, n
        obj, img = found
        # undistort to normalized coordinates, then a plain pinhole PnP
        und = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), self.K,
                                          self.D)
        ok, rvec, tvec = cv2.solvePnP(obj, und, np.eye(3), None)
        if not ok:
            return None, n
        return _tf(tvec, cv2.Rodrigues(rvec)[0]), n

    def project(self, tf_cam_board, obj=None):
        """Pixels of board points ``obj`` (all corners by default) for a
        board-in-camera tf."""
        if obj is None:
            obj = self.board.getChessboardCorners()
        rvec = cv2.Rodrigues(np.ascontiguousarray(tf_cam_board[:3, :3],
                                                  dtype=np.float64))[0]
        tvec = np.ascontiguousarray(tf_cam_board[:3, 3],
                                    dtype=np.float64).reshape(3, 1)
        proj, _ = cv2.fisheye.projectPoints(
            np.ascontiguousarray(obj, dtype=np.float64).reshape(-1, 1, 3),
            rvec, tvec, self.K, self.D)
        return proj.reshape(-1, 2)


def flange_tfs(qs_list):
    """The sampler's flange tf in the world for each joint vector."""
    _, _, sampler = mobisys()
    tfs = []
    for qs in qs_list:
        sampler.fk(qs=np.asarray(qs, dtype=np.float32))
        tfs.append(np.array(sampler.tcp('flange').tf, dtype=np.float64))
    return tfs


def _initial_guess(detector, tfs_w_flange, observations):
    """``(X, W)`` from OpenCV's closed-form hand-eye (Park) and the mean
    board pose it implies."""
    tfs_cam_board = []
    for obj, img in observations:
        und = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), detector.K,
                                          detector.D)
        _, rvec, tvec = cv2.solvePnP(obj, und, np.eye(3), None)
        tfs_cam_board.append(_tf(tvec, cv2.Rodrigues(rvec)[0]))
    rot, pos = cv2.calibrateHandEye(
        [t[:3, :3] for t in tfs_w_flange], [t[:3, 3] for t in tfs_w_flange],
        [t[:3, :3] for t in tfs_cam_board], [t[:3, 3] for t in tfs_cam_board],
        method=cv2.CALIB_HAND_EYE_PARK)
    x = _tf(pos, rot)
    boards = [w @ x @ b for w, b in zip(tfs_w_flange, tfs_cam_board)]
    return x, boards[0]


def solve(detector, tfs_w_flange, observations):
    """``(X flange -> camera, W board in world, per-image RMS px)`` fitted
    to all detected corners. ``observations``: ``(board points, pixels)``
    per image, as from :meth:`BoardDetector.corners`."""
    x0, w0 = _initial_guess(detector, tfs_w_flange, observations)
    inv_flanges = [np.linalg.inv(f) for f in tfs_w_flange]

    def residuals(p):
        inv_x = np.linalg.inv(_tf_from_vec(p[:6]))
        w = _tf_from_vec(p[6:])
        return np.concatenate([
            (detector.project(inv_x @ inv_f @ w, obj) - img).ravel()
            for inv_f, (obj, img) in zip(inv_flanges, observations)])

    p0 = np.concatenate([_vec_from_tf(x0), _vec_from_tf(w0)])
    fit = least_squares(residuals, p0, loss='huber', f_scale=HUBER_PX,
                        x_scale='jac')
    x, w = _tf_from_vec(fit.x[:6]), _tf_from_vec(fit.x[6:])
    res = residuals(fit.x).reshape(-1, 2)
    sizes = np.cumsum([0] + [len(img) for _, img in observations])
    per_image = [float(np.sqrt(np.mean(np.sum(res[a:b] ** 2, axis=1))))
                 for a, b in zip(sizes[:-1], sizes[1:])]
    moved = np.linalg.norm(x[:3, 3] - x0[:3, 3])
    print(f'  fit moved the camera {moved * 1000:.2f} mm from the '
          'closed-form start')
    return x, w, per_image


def report(tf_flange_cam, tf_w_board, per_image, names=None):
    """Print the result and the board check."""
    pos = tf_flange_cam[:3, 3]
    rpy = np.degrees(oum.euler_from_rotmat(tf_flange_cam[:3, :3]))
    print(f'camera in the flange frame: pos {np.round(pos * 1000, 2)} mm, '
          f'rpy {np.round(rpy, 2)} deg')
    errors = np.array(per_image)
    print(f'reprojection error: RMS {np.sqrt(np.mean(errors ** 2)):.3f} px '
          f'over {len(errors)} images, worst image {errors.max():.3f} px')
    if names is not None:
        print('  worst: ' + ', '.join(
            f'{names[i]} {errors[i]:.2f}px'
            for i in np.argsort(errors)[::-1][:N_WORST]))
    tilt = np.degrees(np.arccos(min(abs(float(tf_w_board[2, 2])), 1.0)))
    print(f'board origin in the world: '
          f'{np.round(tf_w_board[:3, 3] * 1000, 2)} mm, tilt from horizontal '
          f'{tilt:.2f} deg (it lies flat at about the base height, so the '
          'tilt should be ~0)')


def main():
    with open(RECORD_PATH, encoding='utf-8') as f:
        records = json.load(f)
    detector = BoardDetector()
    names, qs_list, observations = [], [], []
    for rec in records:
        found, n = detector.corners(
            cv2.imread(os.path.join(DATA_DIR, rec['image'])))
        if found is None:
            print(f"{rec['image']}: only {n} corners, skipped")
            continue
        names.append(rec['image'])
        qs_list.append(rec['qs'])
        observations.append(found)
    if len(qs_list) < 5:
        print(f'only {len(qs_list)} usable poses; need more')
        return False
    print(f'{len(qs_list)} poses, '
          f'{sum(len(img) for _, img in observations)} corners')
    tf_flange_cam, tf_w_board, per_image = solve(
        detector, flange_tfs(qs_list), observations)
    report(tf_flange_cam, tf_w_board, per_image, names)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'tf_flange_cam': tf_flange_cam.tolist(),
                   'tf_w_board': tf_w_board.tolist(),
                   'rms_px': float(np.sqrt(np.mean(np.square(per_image)))),
                   'n_poses': len(qs_list)}, f, indent=2)
    print(OUT_PATH)
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
