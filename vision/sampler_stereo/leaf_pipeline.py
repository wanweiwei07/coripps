"""Look at the plant with the sampler's stereo pair, find the leaves, pick
one, and plan the cutter's tcp onto it -- the library of
``coripps/tool/leaf/suc_cut_store.py``.

* the eye (:class:`Eye`): both webcams kept open at one exposure, a frame
  = the rectified left picture + its depth (rows re-aligned) + K
  (``depth.rgbd``); the hand-eye is the left camera's, carried to the
  rectified left frame (:func:`load_handeye`)
* the eye always looks straight down (its z along the world's -z, the
  roll about the vertical free), and the cutter -- fixed in the picture
  (:func:`tool_mask`) -- kept off the plant: the plant is put at the
  pixel nearest the centre where it stays clear of it (:func:`free_pixel`)
* search: the eye SEARCH_HEIGHT over SEARCH_XY (:func:`search_goal`);
  SAM 3 finds the plants and their leaves, the one nearest SEARCH_XY is
  taken: the circle round its leaves, its top from their depth
  (:func:`find_plant`) -- not the pot: its holder, a label or the cutter
  in front change its outline, and the plant is what must be seen
* observe: the eye OBSERVE_HEIGHT over the plant's top
  (:func:`observe_goal`)
* leaves (``leaves``): SAM 3 finds the plant and its leaves in both
  eyes, each leaf matched on its own (``depth.leaf_depth``); only the
  leaves the right eye sees as the left does are kept (:func:`_found`) --
  the two may cut the rest differently (two leaves as one)
  (``depth.leaf_depth``); a frame per leaf, the leaf with the most room at
  its petiole (:func:`look`)
* closer: the tcp turned to the chosen leaf's J -> C about the vertical
  first, kept level (the pitch, unsure from afar, is the second look's),
  then only shifted: the eye RELOOK_HEIGHT over the plant's top, the leaf
  (C, J and the room round them) clear of the cutter in the picture
  (:func:`relook_goal`) -- the take after it a shift, the pitch and a
  descent, no big turn
* look again from there: the leaf whose C is nearest the old C, refused
  past RELOOK_MAX_MOVE (another leaf) (:func:`relook`)
* onto the leaf: straight to RELOOK_OFFSET over the new C, then straight
  down along the leaf's z to APPROACH_OFFSET (:func:`to_leaf`)

Frames are saved to ``observe/`` (``<prefix>color.png``, ``depth.npy``,
``observe.json``, ``leaves.png``). The collision model is whatever
``pln_ctx`` holds -- NOT the plant or its stand.
"""

import os
import json
import time

import cv2
import numpy as np

import one.geom.loader as ogl
import one.motion.probabilistic.rrt as ompr
import one.motion.probabilistic.post_processor as omppp
import one.utils.math as oum

from coripps.end_effectors.xh430_cutter import xh430_cutter as xc
from coripps.tool import pose_sequence as pseq
from coripps.vision.sampler_stereo import calib
from coripps.vision.sampler_stereo import depth as sd
from coripps.vision.sampler_stereo import leaves as lv
from coripps.vision.sampler_stereo.camera import StereoCam


_HERE = os.path.dirname(os.path.abspath(__file__))
OBSERVE_DIR = os.path.join(_HERE, 'observe')
CUTTER_MESH = os.path.join(os.path.dirname(xc.__file__), 'meshes',
                           'base_stereo.stl')
# the search: the eye looks straight down SEARCH_HEIGHT over SEARCH_XY at
# NOMINAL_PLANT_Z (world, m), a plant up to NOMINAL_PLANT_RADIUS; of the
# plants seen, the one nearest SEARCH_XY
SEARCH_XY = (0.350, -0.190)
NOMINAL_PLANT_Z = 0.055
NOMINAL_PLANT_RADIUS = 0.035
SEARCH_HEIGHT = 0.200
# the eye OBSERVE_HEIGHT over the plant's top (m): the cutter's tube reaches
# ~10 cm past the cameras, so its tip stays ~4 cm over it
OBSERVE_HEIGHT = 0.140
# the plant's top: this percentile of its leaves' depth points' z
PLANT_PCT = 90
MIN_PLANT_PTS = 50
# the cutter in the picture: its mesh projected, grown TOOL_MARGIN_PX (the
# fibre is not in the mesh); the plant is put where it stays
# PLANT_MARGIN_PX clear of that and of the picture's edge
TOOL_MARGIN_PX = 15
PLANT_MARGIN_PX = 10
MIN_TOOL_Z = 0.005              # m, mesh nearer the eye left out
ROLL_STEP_DEG = 15
# the second look, closer: the eye RELOOK_HEIGHT over the plant's top (the
# cutter's tip ~2 cm over it), the leaf's C - J span plus LEAF_VIEW_MARGIN
# each way clear of the cutter; the leaf found again within RELOOK_MAX_MOVE
RELOOK_HEIGHT = 0.120
LEAF_VIEW_MARGIN = 0.025                # m
RELOOK_MAX_MOVE = 0.010                 # m
RELOOK_OFFSET = 0.020                   # m, back along the leaf's z, over C
APPROACH_OFFSET = 0.010                 # m, ditto, where the descent ends
DESCENT_STEP = 0.002                    # m between descent samples
TURN_STEP = np.deg2rad(2.0)             # rad, most turn between samples
ARRIVAL_TOLERANCE = np.deg2rad(0.5)
SETTLE = 0.5                            # s still before a frame
# the depth pair's two pictures may differ this much (brightness, sharpness;
# both past it to stop) -- 2.9-3.2x / 4.5-5.1x seen with a cable over a
# lens, up to 2.1x / 2.8x as the two eyes just see the cutter and the board
# differently (the leaves themselves are checked by _found); a cable
# still swinging just after a move: taken again, PAIR_TRIES times
# PAIR_RETRY_WAIT apart
PAIR_MAX_DIM = 2.5
PAIR_MAX_BLUR = 3.5
PAIR_TRIES = 3
PAIR_RETRY_WAIT = 1.0                   # s
# a leaf is kept only when the right eye sees it too: its left mask, moved
# by the disparity of its depth, overlaps one of the right's leaves by this
# IoU; the right's plant looked for PLANT_DISP_GUESS px over when the left
# has no depth
LEAF_BOTH_IOU = 0.5
PLANT_DISP_GUESS = 65.0                 # px
_EXTEND_STEP_SIZE = np.pi / 36
_MAX_ITERS = 3000


class Eye:
    """The stereo pair, open until :meth:`close`; frames in the rectified
    left frame."""

    def __init__(self):
        self.rect = sd.rectifier()
        self.K = self.rect.K
        self.size = self.rect.size
        self.cam = StereoCam()
        self.rows = None                # the last row fit (depth.disparity)

    def frame(self):
        """``(color BGR, depth m (0: none), K)``, taken now: one pair at
        the cameras' one exposure (``StereoCam.frames``) -- the colour its
        left, the depth of it, rows re-aligned (``depth.disparity``)."""
        for k in range(PAIR_TRIES):
            self.pair = self.cam.frames()       # kept for save_frame
            try:
                check_pair(*self.pair)
                break
            except RuntimeError as exc:
                if k == PAIR_TRIES - 1:
                    raise
                print(f'{exc} -- again in {PAIR_RETRY_WAIT:.0f} s')
                time.sleep(PAIR_RETRY_WAIT)
        color, depth, K, self.rows, self.matched = sd.rgbd(
            *self.pair, self.rect, rows=self.rows)
        self.last_depth = depth
        return color, depth, K

    def leaf_depth(self, mask):
        """A region's own depth in the last frame (``depth.leaf_depth``)."""
        return sd.leaf_depth(*self.matched, self.rect, mask)

    def close(self):
        self.cam.close()


def check_pair(left, right):
    """Raise RuntimeError when the two pictures differ past PAIR_MAX_DIM in
    brightness and PAIR_MAX_BLUR in sharpness (the variance of the
    Laplacian): something close in front of one lens (a cable) dims and
    blurs that picture -- its depth would be wrong."""
    grey = [cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) for p in (left, right)]
    mean = [max(float(g.mean()), 1.0) for g in grey]
    sharp = [max(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1.0)
             for g in grey]
    dim = max(mean) / min(mean)
    blur = max(sharp) / min(sharp)
    if dim > PAIR_MAX_DIM and blur > PAIR_MAX_BLUR:
        side = 'left' if sharp[0] < sharp[1] else 'right'
        raise RuntimeError(
            f'the {side} picture is {dim:.1f}x darker / {blur:.1f}x blurrier '
            f'than the other -- something in front of its lens?')


def load_handeye():
    """Flange -> the rectified left frame (the frame :class:`Eye` gives)."""
    return calib.load_handeye('left') @ sd.tf_cam_rect(sd.rectifier())


def tf_w_cam(sampler, qs, tf_flange_cam):
    sampler.fk(qs=np.asarray(qs, dtype=np.float32))
    return (np.array(sampler.tcp('flange').tf, dtype=np.float64)
            @ tf_flange_cam)


def nearest_valid(pln_ctx, sols, near):
    valid = [np.asarray(q, np.float32) for q in sols
             if pln_ctx.is_state_valid(np.asarray(q, np.float32))]
    if not valid:
        return None
    return min(valid, key=lambda q: np.max(np.abs(q - near)))


def plan(pln_ctx, start, goal):
    """A collision-free joint path, or None."""
    if pln_ctx.is_motion_valid(start, goal):
        return [start, goal]
    planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx,
                                     extend_step_size=_EXTEND_STEP_SIZE)
    path = planner.solve(start=start, goal=goal, max_iters=_MAX_ITERS)
    if not path:
        return None
    # its detours taken out, as khi_v2's transits
    return omppp.PathPostProcessor(pln_ctx).tighten(path)


# -- the eye's poses -----------------------------------------------------
def tool_mask(sampler, tf_flange_cam, eye, dist):
    """The pixels of the left picture where the cutter hides what lies
    ``dist`` (m) off -- from the left eye, or from the right, the right's
    view carried over by the disparity at ``dist`` (the right eye sees the
    cutter elsewhere, and a leaf it hides there has no depth) -- grown by
    TOOL_MARGIN_PX (bool, ``eye.size`` = (w, h)). Its mesh projected: it is
    fixed to the eyes, the same pixels from any pose."""
    tf_cam_cutter = np.linalg.inv(
        np.array(sampler.tcp('flange').tf, dtype=np.float64)
        @ tf_flange_cam) @ np.array(sampler.cutter.tcp('tcp').parent_lnk.tf,
                                    dtype=np.float64)
    geom = ogl.load_geometry(CUTTER_MESH)
    vs = (np.asarray(geom.vs, dtype=np.float64) @ tf_cam_cutter[:3, :3].T
          + tf_cam_cutter[:3, 3])
    tris = vs[np.asarray(geom.fs)]
    tris = tris[(tris[:, :, 2] > MIN_TOOL_Z).all(axis=1)]
    K, rect = eye.K, eye.rect
    f, b = K[0, 0], rect.baseline * (1 if rect.a_left else -1)
    mask = np.zeros((eye.size[1], eye.size[0]), np.uint8)
    for shift in (0.0, b):              # the left eye, the right eye
        px = np.empty(tris.shape[:2] + (2,))
        px[..., 0] = (f * (tris[..., 0] - shift) / tris[..., 2] + K[0, 2]
                      + f * shift / dist)
        px[..., 1] = f * tris[..., 1] / tris[..., 2] + K[1, 2]
        for tri in np.round(px).astype(np.int32):
            cv2.fillConvexPoly(mask, tri, 1)
    grow = 2 * TOOL_MARGIN_PX + 1
    return cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (grow, grow))) > 0


def free_pixel(mask, radius_px, K):
    """The pixel nearest the picture's centre where a disc of
    ``radius_px`` + PLANT_MARGIN_PX is clear of ``mask`` and inside the
    picture; None when there is none."""
    r = radius_px + PLANT_MARGIN_PX
    clear = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = mask.shape
    vv, uu = np.mgrid[0:h, 0:w]
    ok = ((clear >= r) & (uu >= r) & (uu < w - r) & (vv >= r)
          & (vv < h - r))
    if not ok.any():
        return None
    d = (uu - K[0, 2]) ** 2 + (vv - K[1, 2]) ** 2
    d[~ok] = np.inf
    v, u = np.unravel_index(np.argmin(d), d.shape)
    return np.array([u, v], dtype=np.float64)


def eye_goal(sampler, pln_ctx, tf_flange_cam, target, height, px, K, near):
    """Joints putting the eye ``height`` over ``target`` (world), looking
    straight down (its z along the world's -z), ``target`` at pixel ``px``;
    of the rolls about the vertical the collision-free IK nearest
    ``near``."""
    target = np.asarray(target, dtype=np.float64)
    ray = np.array([(px[0] - K[0, 2]) / K[0, 0] * height,
                    (px[1] - K[1, 2]) / K[1, 1] * height, height])
    tf_cam_flange = np.linalg.inv(tf_flange_cam)
    best = None
    for roll in range(0, 360, ROLL_STEP_DEG):
        a = np.deg2rad(roll)
        x = np.array([np.cos(a), np.sin(a), 0.0])
        z = np.array([0.0, 0.0, -1.0])
        rot = np.column_stack([x, np.cross(z, x), z])
        tf_w_flange = lv.tf(target - rot @ ray, rot) @ tf_cam_flange
        qs = nearest_valid(pln_ctx, sampler.ik(
            tf_w_flange[:3, 3].astype(np.float32),
            tf_w_flange[:3, :3].astype(np.float32), ref_qs=near), near)
        if qs is not None and (best is None or np.max(np.abs(qs - near))
                               < np.max(np.abs(best - near))):
            best = qs
    return best


def search_goal(sampler, pln_ctx, tf_flange_cam, eye, near):
    """Joints for the search: the eye SEARCH_HEIGHT over SEARCH_XY (at
    NOMINAL_PLANT_Z), a NOMINAL_PLANT_RADIUS plant there clear of the
    cutter; None when there is none."""
    mask = tool_mask(sampler, tf_flange_cam, eye, SEARCH_HEIGHT)
    px = free_pixel(mask, NOMINAL_PLANT_RADIUS * eye.K[0, 0] / SEARCH_HEIGHT,
                    eye.K)
    if px is None:
        return None
    return eye_goal(sampler, pln_ctx, tf_flange_cam,
                    [*SEARCH_XY, NOMINAL_PLANT_Z], SEARCH_HEIGHT, px, eye.K,
                    near)


def observe_goal(sampler, pln_ctx, tf_flange_cam, eye, plant, near):
    """Joints for the look at the leaves: the eye OBSERVE_HEIGHT over
    ``plant`` (:func:`find_plant`), the whole plant clear of the cutter;
    None when there is none (said why)."""
    mask = tool_mask(sampler, tf_flange_cam, eye, OBSERVE_HEIGHT)
    px = free_pixel(mask, plant['radius'] * eye.K[0, 0] / OBSERVE_HEIGHT,
                    eye.K)
    if px is None:
        print(f"observe: a {plant['radius'] * 1000:.0f} mm plant does not "
              'fit in the picture beside the cutter')
        return None
    return eye_goal(sampler, pln_ctx, tf_flange_cam, plant['center'],
                    OBSERVE_HEIGHT, px, eye.K, near)


def relook_goal(sampler, pln_ctx, tf_flange_cam, eye, leaf, plant_z, near,
                pitch=0.0):
    """Joints for the second look, the tcp already turned as for the take
    about the vertical (``leaf``'s J -> C), level -- or pitched ``pitch``
    (rad, + its J -> C axis up, the nozzle square to it) -- after it only
    moved: the eye RELOOK_HEIGHT over the plant's top ``plant_z``, the
    leaf (C, J and LEAF_VIEW_MARGIN round them) shifted to where the cutter
    does not hide it in the picture; None when there is none (said
    why)."""
    sampler.fk(qs=near)
    tf_tcp_cam = (np.linalg.inv(np.array(sampler.cutter.tcp('tcp').tf,
                                         dtype=np.float64))
                  @ np.array(sampler.tcp('flange').tf, dtype=np.float64)
                  @ tf_flange_cam)
    # turned to J -> C about the vertical only: the pitch, unsure from afar,
    # is the second look's (taken on the way onto the leaf)
    y = np.array([leaf['tf'][0, 1], leaf['tf'][1, 1], 0.0])
    y /= np.linalg.norm(y)
    x = np.cross([0.0, 0.0, 1.0], y)
    y = np.cos(pitch) * y + np.sin(pitch) * np.array([0.0, 0.0, 1.0])
    tcp_rot = np.column_stack([x, y, np.cross(x, y)])
    rot = tcp_rot @ tf_tcp_cam[:3, :3]
    mid = (leaf['C'] + leaf['J']) / 2
    eye_z = plant_z + RELOOK_HEIGHT
    span = np.linalg.norm(leaf['C'] - leaf['J']) / 2 + LEAF_VIEW_MARGIN
    dist = (eye_z - mid[2]) / -rot[2, 2]             # along the eye's axis
    mask = tool_mask(sampler, tf_flange_cam, eye, dist)
    px = free_pixel(mask, span * eye.K[0, 0] / dist, eye.K)
    if px is None:
        print(f'relook: the leaf ({span * 2000:.0f} mm across) does not fit '
              'in the picture beside the cutter')
        return None
    ray = rot @ np.array([(px[0] - eye.K[0, 2]) / eye.K[0, 0],
                          (px[1] - eye.K[1, 2]) / eye.K[1, 1], 1.0])
    tf_w_flange = (lv.tf(mid - (mid[2] - eye_z) / ray[2] * ray, rot)
                   @ np.linalg.inv(tf_flange_cam))
    qs = nearest_valid(pln_ctx, sampler.ik(
        tf_w_flange[:3, 3].astype(np.float32),
        tf_w_flange[:3, :3].astype(np.float32), ref_qs=near), near)
    if qs is None:
        print('relook: no collision-free IK for the turned, shifted pose')
    return qs


def project(pt, cam_tf, K):
    """Pixel of world point ``pt``."""
    pc = cam_tf[:3, :3].T @ (np.asarray(pt, dtype=np.float64)
                             - cam_tf[:3, 3])
    return pc[:2] / pc[2] * K[0, 0] + K[:2, 2]


# -- the plant -----------------------------------------------------------
def find_plant(eye, sampler, qs, tf_flange_cam, segmenter, prefix='plant_'):
    """A frame from joints ``qs``, saved; of the plants seen (SAM 3,
    ``leaves.plant_mask``), the one nearest SEARCH_XY: ``{'center': the
    centre of the circle round its leaves at its top (world), 'radius' (m),
    'plant_z', 'px', 'r_px'}`` -- the top the PLANT_PCT percentile of its
    leaves' depth points' z (each leaf its own depth) -- or None (none
    seen, or too little depth on it)."""
    color, depth, K = eye.frame()
    save_frame(color, depth, K, qs, prefix, eye.pair)
    cam_tf = tf_w_cam(sampler, qs, tf_flange_cam)
    shown = lv.normalize(color)
    found = lv.plant_mask(shown, segmenter,
                          project([*SEARCH_XY, NOMINAL_PLANT_Z], cam_tf, K))
    if found is None:
        return None
    leaf_masks = found[1]
    pts = [lv.depth_points(m, z, K, cam_tf)
           for m, z in ((m, eye.leaf_depth(m)) for m in leaf_masks)
           if z is not None]
    pts = np.vstack(pts) if pts else np.zeros((0, 3))
    if len(pts) < MIN_PLANT_PTS:
        return None
    plant_z = float(np.percentile(pts[:, 2], PLANT_PCT))
    leaf_px = np.argwhere(np.any(leaf_masks, axis=0))[:, ::-1]
    (cx, cy), r_px = cv2.minEnclosingCircle(leaf_px.astype(np.float32))
    o, v = lv.ray((cx, cy), cam_tf, K)
    center = o + (plant_z - o[2]) / v[2] * v
    radius = r_px * (cam_tf[:3, 2] @ (center - o)) / K[0, 0]
    cv2.circle(shown, (int(round(cx)), int(round(cy))), int(round(r_px)),
               (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OBSERVE_DIR, f'{prefix}found.png'), shown)
    return {'center': center, 'radius': float(radius), 'plant_z': plant_z,
            'px': np.array([cx, cy]), 'r_px': float(r_px)}


def save_frame(color, depth, K, qs, prefix='', pair=None):
    """A frame and where it was taken from, to ``observe/``:
    ``<prefix>color.png``, ``<prefix>depth.npy``, ``<prefix>observe.json``
    (joints, K) -- read back by :func:`load_observation` -- and the raw
    depth ``pair`` (``<prefix>pair_left.png`` / ``_right``), to recheck."""
    os.makedirs(OBSERVE_DIR, exist_ok=True)
    if pair is not None:
        for side, pic in zip(('left', 'right'), pair):
            cv2.imwrite(os.path.join(OBSERVE_DIR, f'{prefix}pair_{side}.png'),
                        pic)
    cv2.imwrite(os.path.join(OBSERVE_DIR, f'{prefix}color.png'), color)
    np.save(os.path.join(OBSERVE_DIR, f'{prefix}depth.npy'), depth)
    with open(os.path.join(OBSERVE_DIR, f'{prefix}observe.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'qs': [float(q) for q in qs], 'K': np.asarray(K).tolist()},
                  f, indent=2)


def load_observation(prefix=''):
    """``(color, depth, K, qs)`` saved by :func:`save_frame`."""
    with open(os.path.join(OBSERVE_DIR, f'{prefix}observe.json'),
              encoding='utf-8') as f:
        rec = json.load(f)
    return (cv2.imread(os.path.join(OBSERVE_DIR, f'{prefix}color.png')),
            np.load(os.path.join(OBSERVE_DIR, f'{prefix}depth.npy')),
            np.array(rec['K']), np.array(rec['qs'], np.float32))


def right_leaves(eye, segmenter, near=None):
    """The right eye's leaves of the plant (SAM 3 on the rectified right of
    the last frame): of the plant a disparity of the left's plant depth over
    from where the left sees it (the one nearest ``near``). Raise
    RuntimeError when either eye sees no plant."""
    rect_l, rect_r = eye.matched
    left = lv.plant_mask(lv.normalize(rect_l), segmenter, near)
    if left is None:
        raise RuntimeError('the left eye sees no plant')
    zs = eye.last_depth[left[0] & (eye.last_depth > 0)]
    d = (eye.rect.f * eye.rect.baseline / np.median(zs) if len(zs)
         else PLANT_DISP_GUESS)
    u, v = np.argwhere(left[0])[:, ::-1].mean(0)
    right = lv.plant_mask(lv.normalize(rect_r), segmenter,
                          (u - (d if eye.rect.a_left else -d), v))
    if right is None:
        raise RuntimeError('the right eye sees no plant')
    return right[1]


def right_iou(eye, mask, z, right):
    """How alike the right eye sees a left leaf ``mask`` at depth ``z``
    (m): the IoU of the mask moved by its disparity with the best of the
    right's leaf masks ``right``."""
    d = eye.rect.f * eye.rect.baseline / z
    h, w = mask.shape
    moved = cv2.warpAffine(
        mask.astype(np.uint8),
        np.float32([[1, 0, -d if eye.rect.a_left else d], [0, 1, 0]]),
        (w, h), flags=cv2.INTER_NEAREST) > 0
    return max(((moved & r).sum() / max((moved | r).sum(), 1)
                for r in right), default=0.0)


def _found(eye, color, depth, K, cam_tf, segmenter, near=None):
    """``lv.leaf_frames``' ``(leaves, rosette)``, only the leaves the right
    eye sees alike (LEAF_BOTH_IOU); None when none."""
    right = right_leaves(eye, segmenter, near)
    found = lv.leaf_frames(color, depth, K, cam_tf, segmenter, near,
                           eye.leaf_depth)
    if found is None:
        return None
    leaves, rosette = found
    kept = []
    for k, leaf in enumerate(leaves):
        z = float(np.median((leaf['blade_pts'] - cam_tf[:3, 3])
                            @ cam_tf[:3, 2]))
        iou = right_iou(eye, leaf['mask'], z, right)
        if iou < LEAF_BOTH_IOU:
            print(f'leaf {k}: the right eye does not see it alike (IoU '
                  f'{iou:.2f}) -- skipped')
            continue
        kept.append(leaf)
    return (kept, rosette) if kept else None


def _save_leaves(color, leaves, rosette, chosen, prefix):
    cv2.imwrite(os.path.join(OBSERVE_DIR, f'{prefix}leaves.png'),
                lv.draw(lv.normalize(color), leaves, rosette, chosen))


def look(eye, sampler, qs, tf_flange_cam, segmenter, plant, prefix=''):
    """A frame from joints ``qs``, saved; of ``plant``
    (:func:`find_plant`), ``(leaves, chosen index)``, or None when no leaf
    is found."""
    color, depth, K = eye.frame()
    save_frame(color, depth, K, qs, prefix, eye.pair)
    cam_tf = tf_w_cam(sampler, qs, tf_flange_cam)
    found = _found(eye, color, depth, K, cam_tf, segmenter,
                   project(plant['center'], cam_tf, K))
    if found is None:
        return None
    leaves, rosette = found
    chosen = lv.choose(leaves)
    _save_leaves(color, leaves, rosette, chosen, prefix)
    return leaves, chosen


def relook(eye, sampler, qs, tf_flange_cam, segmenter, old,
           prefix='relook_'):
    """The leaf ``old`` seen again from joints ``qs``: of the plant round
    ``old``'s C, the leaf whose C is nearest it; the new leaf dict, or None
    when no leaf is found or the nearest moved past RELOOK_MAX_MOVE (said
    why)."""
    color, depth, K = eye.frame()
    save_frame(color, depth, K, qs, prefix, eye.pair)
    cam_tf = tf_w_cam(sampler, qs, tf_flange_cam)
    found = _found(eye, color, depth, K, cam_tf, segmenter,
                   project(old['C'], cam_tf, K))
    if found is None:
        print('relook: no leaves found')
        return None
    leaves, rosette = found
    k = int(np.argmin([np.linalg.norm(leaf['C'] - old['C'])
                       for leaf in leaves]))
    _save_leaves(color, leaves, rosette, k, prefix)
    new = leaves[k]
    turn = np.degrees(np.arccos(np.clip(
        (np.trace(old['tf'][:3, :3].T @ new['tf'][:3, :3]) - 1) / 2, -1, 1)))
    print(f"relook: the leaf is leaf {k} now; C moved "
          f"{np.round((new['C'] - old['C']) * 1000, 2)} mm, J moved "
          f"{np.round((new['J'] - old['J']) * 1000, 2)} mm, frame turned "
          f"{turn:.2f} deg")
    if np.linalg.norm(new['C'] - old['C']) > RELOOK_MAX_MOVE:
        print(f'relook: C moved past {RELOOK_MAX_MOVE * 1000:.0f} mm -- '
              'taken for another leaf')
        return None
    return new


# -- approach ------------------------------------------------------------
def back_along_z(leaf_tf, offset):
    """The leaf frame moved ``offset`` back along its own z."""
    tgt = np.array(leaf_tf, dtype=np.float64)
    tgt[:3, 3] -= offset * tgt[:3, 2]
    return tgt


def straight(sampler, pln_ctx, qs, positions, rotmat, what):
    """A straight tcp path through ``positions`` (``qs`` first), or None
    (said why). ``rotmat``: one orientation for all, or one per position
    (the tcp turning on the way)."""
    rotmats = (rotmat if np.ndim(rotmat) == 3
               else [rotmat] * len(positions))
    path = [np.asarray(qs, np.float32)]
    for pos, rot in zip(positions, rotmats):
        step = pseq.follow_positions(sampler, path[-1],
                                     [np.asarray(pos, np.float32)],
                                     np.asarray(rot, np.float32),
                                     tcp=sampler.cutter.tcp('tcp'))
        if step is None:
            print(f'{what}: no smooth IK')
            return None
        path.append(step[-1])
    if not all(pln_ctx.is_motion_valid(a, b)
               for a, b in zip(path[:-1], path[1:])):
        print(f'{what}: in collision')
        return None
    return list(path)


def to_leaf(sampler, pln_ctx, leaf_tf, qs):
    """Onto the leaf: straight to RELOOK_OFFSET over its C, then straight
    down to APPROACH_OFFSET; one path, or None (said why)."""
    sampler.fk(qs=qs)
    cur_tf = np.array(sampler.cutter.tcp('tcp').tf, dtype=np.float64)
    cur = cur_tf[:3, 3]
    pre = back_along_z(leaf_tf, RELOOK_OFFSET)[:3, 3]
    turn = np.arccos(np.clip((np.trace(cur_tf[:3, :3].T @ leaf_tf[:3, :3])
                              - 1) / 2, -1, 1))
    # the position and the orientation both on the way, evenly: as many
    # samples as the longer of the two needs
    n = max(int(np.ceil(np.linalg.norm(pre - cur) / DESCENT_STEP)),
            int(np.ceil(turn / TURN_STEP)), 1)
    rots = oum.rotmat_slerp(cur_tf[:3, :3], leaf_tf[:3, :3], n + 1)[1:]
    over = straight(sampler, pln_ctx, qs,
                    [cur + (pre - cur) * k / n for k in range(1, n + 1)],
                    rots, 'over the leaf')
    if over is None:
        return None
    n = int(round((RELOOK_OFFSET - APPROACH_OFFSET) / DESCENT_STEP))
    down = straight(sampler, pln_ctx, over[-1],
                    [back_along_z(leaf_tf, RELOOK_OFFSET - (
                        RELOOK_OFFSET - APPROACH_OFFSET) * k / n)[:3, 3]
                     for k in range(1, n + 1)],
                    leaf_tf[:3, :3], 'down to the leaf')
    return None if down is None else over + down[1:]
