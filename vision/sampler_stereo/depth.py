"""Depth from the sampler's two webcams (``one.vision.stereo_depth``):
rectified with the stereo calibration (``calib.STEREO_PATH``), the rows
re-aligned per pair (the two webcams drift after calibration: 0.4-1.6 px
seen, enough to warp a pot's rim by mm -- ``StereoRectifier.row_offset``),
SGBM disparity, points in the LEFT camera's frame -- and, with its
hand-eye and the sampler's joint angles, in the world.

Board check (40 pairs at 0.12-0.26 m): rectified rows agree to 0.06 px
(median), the depth on the board is off its plane by 0.2 mm (median
bias), sigma 1.2 mm, 0.4 % outliers past 1 cm.
"""

import cv2
import numpy as np

from one.vision import stereo_depth as osd

from coripps.vision.sampler_stereo import calib


RECT_F = 300.0                  # px, rectified focal (~ the fisheyes', 298)
DEPTH_MIN, DEPTH_MAX = 0.05, 0.5    # m, kept depth range
# leaf_depth: a region's own disparity found whole first (score at least
# LEAF_MIN_SCORE, searched up to LEAF_MAX_DISP, LEAF_ROWS rows either way),
# then per pixel only within LEAF_DISP_SPAN of it
LEAF_MIN_SCORE = 0.7
LEAF_MAX_DISP = 128             # px
LEAF_ROWS = 2                   # px
LEAF_DISP_SPAN = 16             # px, either way (2x: a multiple of 16)
LEAF_KEEP = 4.0                 # px, a pixel's own disparity kept within this
                                # of the whole's (~7 mm at 0.12 m; a leaf's
                                # own depth span, a mismatch's not)
LEAF_BAND_PAD = 8               # px, rows round the region given to SGBM
STRIDE = 2                      # every STRIDE-th pixel in the world view
UPDATE_INTERVAL = 0.3           # s, world view
FRAME_LENGTH = 0.05             # m, drawn camera frame
WINDOW = 'sampler stereo depth (q quits)'


def rectifier():
    """The :class:`StereoRectifier` of left (a) and right (b)."""
    return osd.StereoRectifier(*calib.load_intrinsics('left'),
                               *calib.load_intrinsics('right'),
                               calib.load_stereo(), f=RECT_F)


def disparity(left, right, rect, rows=None):
    """``(disparity of the rectified left, rectified left, rectified right
    (rows put on the left's), rows)`` of one picture pair, the right's rows
    put on the left's first: by the offset fitted to this pair, else by
    ``rows`` (the last fit; said so), else as calibrated (said so).
    ``rows``: the fit used, for the next pair."""
    rect_l, rect_r = rect.rectify(left, right)
    # the two cameras are not alike in brightness at one exposure (~1.3x):
    # the right scaled to the left's over the pictures' middles (SGBM
    # matches brightness; the rows' SIFT and the leaves' whole match do not
    # mind)
    h, w = rect_l.shape[:2]
    mid = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    gain = rect_l[mid].mean() / max(rect_r[mid].mean(), 1.0)
    rect_r = np.clip(rect_r * gain, 0, 255).astype(np.uint8)
    fit, n = rect.row_offset(rect_l, rect_r)
    if fit is None:
        print(f'stereo rows: no fit ({n} matches) -- '
              + ('the last one kept' if rows is not None else 'as calibrated'))
        fit = rows
    else:
        w, h = rect.size
        at = [fit @ (1, u, v) for u, v in ((0, 0), (w, 0), (0, h), (w, h))]
        print(f'stereo rows: {fit[0]:+.2f} px at the origin, '
              f'{min(at):+.2f} .. {max(at):+.2f} at the corners ({n} matches)')
    if fit is not None:
        rect_r = rect.align_rows(rect_r, fit)
    return rect.disparity(rect_l, rect_r), rect_l, rect_r, fit


def cloud(left, right, rect, stride=1, rows=None):
    """``(points (n, 3) in the left camera's frame, BGR colours (n, 3),
    disparity, rectified left, rows)`` of one picture pair (``rows``: as
    :func:`disparity`'s)."""
    disp, rect_l, _, rows = disparity(left, right, rect, rows)
    z = rect.depth(disp)
    keep = (z > DEPTH_MIN) & (z < DEPTH_MAX)
    if stride > 1:
        sub = np.zeros_like(keep)
        sub[::stride, ::stride] = True
        keep &= sub
    pts, (v, u) = rect.points(disp, keep)
    return pts, rect_l[v, u], disp, rect_l, rows


def rgbd(left, right, rect, color=None, rows=None):
    """``(rectified colour BGR, depth (m, 0 where none), K, rows, the
    rectified pair)`` -- one pinhole frame for both, the rectified left
    (:func:`tf_cam_rect`): the depth of the pair ``left`` / ``right``
    (:func:`disparity`, ``rows`` as its; the pair as it matched, for
    :func:`leaf_depth`), the colour of ``color`` (a left picture taken from
    the same pose, e.g. exposure-fused; default ``left``)."""
    disp, rect_l, rect_r, rows = disparity(left, right, rect, rows)
    z = rect.depth(disp)
    z[~((z > DEPTH_MIN) & (z < DEPTH_MAX))] = 0.0
    pair = (rect_l, rect_r)
    if color is not None:
        rect_l = rect.rectify(color, color)[0]
    return rect_l, z, rect.K.copy(), rows, pair


def leaf_depth(rect_l, rect_r, rect, mask):
    """Depth (m, 0 off it) of one region of the rectified left -- a leaf's
    mask -- from the pair as matched (:func:`rgbd`'s): its disparity found
    whole first (its box template-matched along its rows in the right),
    then SGBM per pixel only within LEAF_DISP_SPAN of that, so the ground
    behind a thin leaf cannot pull it away (whole-picture SGBM put such a
    leaf at the board 90 mm below), and of those only the pixels within
    LEAF_KEEP of it kept (the rest stuck at the span's edge: mismatches).
    None when the whole match is poor."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    grey_l, grey_r = (cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3
                      else p for p in (rect_l, rect_r))
    lo, hi = max(y0 - LEAF_ROWS, 0), min(y1 + LEAF_ROWS, h)
    if rect.a_left:                     # b sees it further left
        wx0, wx1 = max(x0 - LEAF_MAX_DISP, 0), x1
    else:
        wx0, wx1 = x0, min(x1 + LEAF_MAX_DISP, w)
    if hi - lo < y1 - y0 or wx1 - wx0 < x1 - x0:
        return None
    score = cv2.matchTemplate(grey_r[lo:hi, wx0:wx1], grey_l[y0:y1, x0:x1],
                              cv2.TM_CCOEFF_NORMED)
    _, best, _, (bx, _) = cv2.minMaxLoc(score)
    if best < LEAF_MIN_SCORE:
        return None
    d0 = abs(wx0 + bx - x0)
    band = slice(max(y0 - LEAF_BAND_PAD, 0), min(y1 + LEAF_BAND_PAD, h))
    lo_d = max(d0 - LEAF_DISP_SPAN, 0)

    def matcher(a, b):
        return osd.disparity_sgbm(a, b, num_disparities=2 * LEAF_DISP_SPAN,
                                  min_disparity=lo_d)
    disp = np.full(mask.shape, np.nan, np.float32)
    disp[band] = rect.disparity(rect_l[band], rect_r[band], matcher)
    disp[~(np.abs(disp - d0) <= LEAF_KEEP)] = np.nan
    z = rect.depth(disp)
    z[~(mask & np.isfinite(z))] = 0.0
    return z


def tf_cam_rect(rect):
    """The rectified left frame's 4x4 pose in the left camera's frame
    (``hand-eye @ tf_cam_rect``: flange -> the frame :func:`rgbd` is in)."""
    tf = np.eye(4)
    tf[:3, :3] = rect.R_a.T
    return tf


def _disparity_image(disp, rect):
    """Disparity coloured over the working range (near red, far blue)."""
    d_far = rect.f * rect.baseline / DEPTH_MAX
    d_near = rect.f * rect.baseline / DEPTH_MIN
    scaled = np.clip((np.nan_to_num(disp) - d_far) / (d_near - d_far), 0, 1)
    shown = cv2.applyColorMap((scaled * 255).astype(np.uint8),
                              cv2.COLORMAP_JET)
    shown[~np.isfinite(disp)] = 0
    return shown


def live():
    """The left picture as the segmenter gets it (rectified,
    ``leaves.normalize``) | the pair's disparity (rows re-aligned), live,
    until q / Esc; p switches the gamma + CLAHE of ``normalize`` on for any
    picture (forced) / back."""
    import time
    from coripps.vision.sampler_stereo.camera import StereoCam
    from coripps.vision.sampler_stereo.leaves import normalize
    rect = rectifier()
    try:
        cam = StereoCam()
    except RuntimeError as exc:
        print(exc)
        return False
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        forced = False
        rows = None
        while True:
            t = time.perf_counter()
            try:
                pair = cam.frames(now=False)
            except RuntimeError as exc:
                print(exc)
                return False
            disp, rect_l, _, rows = disparity(*pair, rect, rows)
            shown = normalize(rect_l, force=forced).copy()
            text = (f'rows {0.0 if rows is None else rows[0]:+.2f} px  '
                    f'{time.perf_counter() - t:.1f} s  '
                    + ('gamma+CLAHE forced' if forced else 'as segmented'))
            cv2.putText(shown, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            cv2.imshow(WINDOW, np.hstack([shown,
                                          _disparity_image(disp, rect)]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                return True
            if key == ord('p'):
                forced = not forced
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return True
    finally:
        cam.close()
        cv2.destroyAllWindows()


def world():
    """The robots at their real joint angles and the stereo cloud put in
    the world with the sampler's FK and the left hand-eye. Jog the
    sampler: with a good calibration the cloud of the table stays put.
    Read-only -- nothing is sent to the robots."""
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw
    from one.drivers.cobotta import BCapError
    from coripps.robots.mobisys import mobisys
    from coripps.tool.home_sampler import _SAMPLER_HOST, _PICKER_HOST
    from coripps.vision.d405.check_pointcloud import connect
    from coripps.vision.sampler_stereo.camera import StereoCam

    rect = rectifier()
    tf_flange_cam = calib.load_handeye('left')
    cam = StereoCam()
    base = ovw.World(cam_pos=(1.0, 0.2, 0.7),
                     cam_lookat_pos=(0.35, -0.15, 0.10))
    mobisys_base, picker, sampler = mobisys()
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    links = [(picker, connect(_PICKER_HOST, 'picker')),
             (sampler, connect(_SAMPLER_HOST, 'sampler'))]
    shown = {'cloud': None, 'frame': None, 'rows': None}

    def update(dt):
        for robot, cobotta in links:
            if cobotta is not None:
                try:
                    robot.fk(qs=cobotta.joint_angles(ndof=robot.ndof))
                except (OSError, BCapError):
                    pass
        try:
            pics = cam.frames()
        except RuntimeError:
            return
        pts, bgr, _, _, shown['rows'] = cloud(*pics, rect, stride=STRIDE,
                                              rows=shown['rows'])
        tf_w_cam = (np.array(sampler.tcp('flange').tf, dtype=np.float64)
                    @ tf_flange_cam)
        pts = pts @ tf_w_cam[:3, :3].T + tf_w_cam[:3, 3]
        for name in ('cloud', 'frame'):
            if shown[name] is not None:
                shown[name].detach_from(base.scene)
        shown['cloud'] = ossop.point_cloud(pts.astype(np.float32),
                                           bgr[:, ::-1] / 255.0)
        shown['cloud'].attach_to(base.scene)
        shown['frame'] = ossop.frame_from_tf(
            tf_w_cam, length_scale=FRAME_LENGTH / 0.2,
            radius_scale=FRAME_LENGTH / 0.2)
        shown['frame'].attach_to(base.scene)

    base.schedule_interval(update, interval=UPDATE_INTERVAL)
    try:
        base.run()
    finally:
        cam.close()
        for _, cobotta in links:
            if cobotta is not None:
                cobotta.disconnect()
    return True
