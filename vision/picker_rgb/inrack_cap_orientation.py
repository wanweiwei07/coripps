"""Which way the tube cap's long tab points, seen by the picker's webcam.

The cap is a round body with a tab on two sides: on one a thin flat strap,
on the other a wide boxed lump -- the long tab (brim), which has to point to
-y. The camera on the picker looks down at the cap from the tube's
pre_pick (straight above it, ``tool/tube/layout_pick``), the gripper opened
for the camera (wider) to keep the fingers out of the picture.

1. top view: where the camera is comes from the joint angles (FK of the
   flange) and the hand-eye calibration (``calib.py``); the plane of the cap
   top (Z_CAP, world) around the tube's xy is re-sampled from the picture
   into a metric, world-aligned top view (MM_PX per pixel, +x right, +y
   up). Lens distortion and perspective are gone: the cap is a circle of
   known size at a known place, for every tube and either TCP orientation.
2. the cap and its turn, by the cap's template -- the mean silhouette of
   several clean caps (:func:`build_template`, ``cap_template.npz``, the
   long tab at 0 deg). SAM 3's point prompt (``one.vision.sam3``) at the
   top view's centre (the tube's xy, the cap is within a few mm of it)
   gives 3 candidates of "the thing clicked"; on each (its piece over the
   centre) the template is turned through 360 deg and slid around (IoU).
   The best fit is the cap, its turn the long tab's direction. The whole
   shape decides, so a ragged edge or a bit of leak costs every turn
   alike.

   No candidate fits (MIN_IOU): no cap. The others are a part of the cap
   (the centre ring), a leak into the rack, or, over an empty hole, the
   ring around it -- caps fit 0.90-0.95, all else 0.41 at most (0.72 a
   ring with noise). The turn is trusted when it beats the best turn with
   the long tab the other way round (+-45 deg about the opposite) by
   MIN_MARGIN -- swapping the long tab and the strap is the one real
   mix-up.

Angles are in the world, degrees, 0 = +x, 90 = +y (so -y is 270).

Run, a menu:

1. live -- the picker goes tube by tube (``config/poses_pick.json``'s
   order) to each pre_pick, takes a picture and finds the cap, until a
   tube has no cap (or the last tube, or q / Esc). Each picture is saved as
   a sample.
2. saved samples -- ``samples/*.png``, each with its ``.json`` (the joint
   angles and the tube's xy, as :func:`save_sample` writes them); nothing
   moves. Any key: next, q / Esc / closing the window: stop.
3. template -- the picker goes tube by tube as in 1, until a tube has no
   cap; each cap SAM 3's text and point prompts agree on
   (:func:`clean_mask`) goes into the template (:func:`build_template`).
   Needs no template; again when the caps change. Its pictures are saved as
   samples ``template_tube<k>``.

    py -3.12 coripps/vision/picker_rgb/inrack_cap_orientation.py
"""

import os
import sys
import glob
import json

import cv2
import numpy as np

from coripps.vision.picker_rgb import calib


Z_CAP = 0.062                   # m (world), the cap top of a tube in the rack
VIEW_HALF = 0.025               # m, the top view spans +-this around the tube
MM_PX = 0.1                     # mm per top-view pixel
MATCH_MM_PX = 0.25              # mm per pixel, matching the template
TMPL_HALF_MM = 16               # mm, the template spans +-this (the cap:
                                # body r ~9 + long tab ~4.5)
MATCH_SHIFT_MM = 4              # mm, the template slides this far
PROMPT = 'translucent plastic cap'  # the template scan's text prompt: the
                                # lid, the thin strap left out
MIN_LID_COVER = 0.99            # a clean cap: a point candidate covers the
LID_RATIO = (1.00, 1.10)        # lid, its area this many lids (+ the strap,
                                # ~5%: 1.02-1.06; a leak: 1.27-2.67)
MIN_TEMPLATE_CAPS = 5           # clean caps a template is made of, at least
MIN_IOU = 0.8                   # the fit at least this (caps: 0.90-0.95,
                                # no cap: 0.72 at most)
MIN_MARGIN = 0.02               # ... and this much better than the long
                                # tab the other way round. At most ~0.1:
                                # (box - strap) / cap area ~ (48 - 15) /
                                # 320 mm2; caps: 0.03-0.09
N_AVG = 10                      # frames averaged per picture (noise)
LIVE_SPEED = 30.0               # percent, the arm between the tubes
WINDOW = 'cap orientation'
_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(_HERE, 'samples')
TEMPLATE_PATH = os.path.join(_HERE, 'cap_template.npz')

_segmenter = None


def segmenter():
    """SAM 3's point prompt (``PointSegmenter``), loaded on first use."""
    global _segmenter
    if _segmenter is None:
        from one.vision.sam3 import PointSegmenter
        _segmenter = PointSegmenter()
    return _segmenter


# ---- picture -> top view ---------------------------------------------------
def camera_tf(robot, qs):
    """4x4 camera pose in the world for the picker's joints ``qs``."""
    robot.fk(qs=np.asarray(qs, dtype=np.float32))
    return (np.array(robot.tcp('flange').tf, dtype=np.float64)
            @ calib.load_handeye())


def top_view(img, tf_w_cam, center_xy, z=Z_CAP, intrinsics=None):
    """The plane ``z`` around ``center_xy`` (world, m) as seen in ``img``:
    a square image, MM_PX per pixel, +x right, +y up, ``center_xy`` in its
    middle."""
    K, D = calib.load_intrinsics() if intrinsics is None else intrinsics
    n = int(round(2 * VIEW_HALF * 1000 / MM_PX))
    s = (np.arange(n) + 0.5) * MM_PX / 1000 - VIEW_HALF
    gx, gy = np.meshgrid(s, -s)            # row 0 is +y
    pts = np.stack([center_xy[0] + gx, center_xy[1] + gy,
                    np.full_like(gx, z)], -1).reshape(-1, 3)
    tf_cam_w = np.linalg.inv(tf_w_cam)
    pc = pts @ tf_cam_w[:3, :3].T + tf_cam_w[:3, 3]
    px, _ = cv2.fisheye.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3),
                                      np.zeros(3), K, D)
    mp = px.reshape(n, n, 2).astype(np.float32)
    return cv2.remap(img, mp[..., 0], mp[..., 1], cv2.INTER_LINEAR)


# ---- the cap in the top view -----------------------------------------------
def candidates(top, seg=None):
    """SAM 3's masks of "the thing at the top view's centre" (bool), each
    only its piece over the centre (drops bits of the neighbours)."""
    seg = segmenter() if seg is None else seg
    n = top.shape[0]
    c = n // 2
    found, _ = seg.masks(top, (n / 2, n / 2))
    pieces = []
    for mask in found:
        if mask[c, c]:
            _, lab = cv2.connectedComponents(mask.astype(np.uint8))
            pieces.append(lab == lab[c, c])
    return pieces


def best_fit(smalls, tmpl):
    """Of the small masks (:func:`_small`), the one the template fits best:
    ``(k, iou (360,), centre (360, 2))`` (see :func:`match`), ``k`` its
    index; ``(None, None, None)`` without masks."""
    best = (None, None, None)
    for k, m in enumerate(smalls):
        iou, centre = match(m, tmpl)
        if best[0] is None or iou.max() > best[1].max():
            best = (k, iou, centre)
    return best


def _small(mask):
    """A top-view mask at MATCH_MM_PX per pixel, float (edges soft)."""
    k = MM_PX / MATCH_MM_PX
    return cv2.resize(mask.astype(np.float32), None, fx=k, fy=k,
                      interpolation=cv2.INTER_AREA)


def _turn(img, deg, centre=None, to=None, size=None):
    """``img`` turned by ``deg`` (counter-clockwise as seen, i.e. in the
    world) about ``centre`` (default its middle), that point moved to
    ``to`` (default the same place), on a ``size`` square (default its
    own)."""
    h = img.shape[0] if size is None else size
    mid = ((img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2)
    centre = mid if centre is None else centre
    to = centre if to is None else to
    rot = cv2.getRotationMatrix2D(tuple(map(float, centre)), deg, 1.0)
    rot[:, 2] += (to[0] - centre[0], to[1] - centre[1])
    return cv2.warpAffine(img, rot, (h, h), flags=cv2.INTER_LINEAR)


def match(m, tmpl):
    """How well ``tmpl`` (h x h, the cap's body centre in its middle),
    turned by each whole degree, covers the small mask ``m`` at its best
    shift (within MATCH_SHIFT_MM of where ``m`` lies): ``(iou (360,),
    centre (360, 2))``, ``centre`` the template's middle in ``m``'s pixels
    (u, v)."""
    h = tmpl.shape[0]
    s = int(round(MATCH_SHIFT_MM / MATCH_MM_PX))
    vs, us = np.nonzero(m > 0.5)
    pad = h + s
    mp = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                            value=0)
    u0 = int(round(us.mean())) - h // 2 - s     # window corner, m's px
    v0 = int(round(vs.mean())) - h // 2 - s
    win = mp[v0 + pad:v0 + pad + h + 2 * s, u0 + pad:u0 + pad + h + 2 * s]
    m_sum = float(m.sum())
    iou = np.zeros(360)
    centre = np.zeros((360, 2))
    for deg in range(360):
        t = _turn(tmpl, deg)
        cc = cv2.matchTemplate(win, t, cv2.TM_CCORR)
        score = cc / (float(t.sum()) + m_sum - cc)
        dv, du = np.unravel_index(np.argmax(score), score.shape)
        iou[deg] = score[dv, du]
        centre[deg] = (u0 + du + (h - 1) / 2, v0 + dv + (h - 1) / 2)
    return iou, centre


def clean_mask(top, text, point):
    """The cap in ``top`` when the text and point prompts agree on it, for
    the template: ``(mask, why)``, ``mask`` None when not clean. The text
    prompt's lid over the centre; of the point candidates, the largest
    covering it (MIN_LID_COVER) at LID_RATIO times its area -- the lid and
    the strap. ``why`` 'no cap' when there is no lid."""
    c = top.shape[0] // 2
    lids = [m for m in text.masks(top, PROMPT) if m[c, c]]
    if not lids:
        return None, 'no cap'
    lid = lids[0]
    best, rows = None, []
    for p in candidates(top, point):
        cover = (p & lid).sum() / lid.sum()
        ratio = p.sum() / lid.sum()
        rows.append(f'{cover:.2f}/{ratio:.2f}')
        if (cover >= MIN_LID_COVER and LID_RATIO[0] <= ratio <= LID_RATIO[1]
                and (best is None or p.sum() > best.sum())):
            best = p
    return best, 'cover/ratio ' + ' '.join(rows)


def build_template(masks, names=(), n_iter=3, path=TEMPLATE_PATH):
    """The cap's template from clean cap masks (top view, bool): turned and
    shifted onto the mean of the last round (the first onto the first
    mask), and averaged -- a soft silhouette, the share of the caps
    covering each pixel. Turned so the long tab is at 0 deg: it is the
    bigger side, so the silhouette's centre of mass lies off the body
    centre (its largest inscribed circle) towards it. Saved to ``path``
    (and a .png to look at)."""
    h = 2 * int(round(TMPL_HALF_MM / MATCH_MM_PX))
    mid = ((h - 1) / 2, (h - 1) / 2)
    caps = []                       # on the canvas, their mean in the middle
    for mask in masks:
        sm = _small(mask)
        vs, us = np.nonzero(sm > 0.5)
        caps.append(_turn(sm, 0, (us.mean(), vs.mean()), mid, h))
    ref = caps[0]
    for it in range(n_iter):
        aligned, ious = [], []
        for m in caps:
            iou, centre = match(m, ref)
            deg = int(np.argmax(iou))
            aligned.append(_turn(m, -deg, centre[deg], mid))
            ious.append(iou[deg])
        ref = np.mean(aligned, 0)
        print(f'round {it + 1}: iou to the mean '
              + ' '.join(f'{v:.2f}' for v in ious))
    dt = cv2.distanceTransform((ref > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    bv, bu = np.unravel_index(np.argmax(dt), dt.shape)
    vs, us = np.mgrid[:h, :h]
    gu, gv = (ref * us).sum() / ref.sum(), (ref * vs).sum() / ref.sum()
    long_deg = np.degrees(np.arctan2(-(gv - bv), gu - bu))
    tmpl = _turn(ref, -long_deg, (float(bu), float(bv)), mid)
    np.savez(path, tmpl=tmpl.astype(np.float32), mm_px=MATCH_MM_PX,
             samples=np.array(list(names)))
    cv2.imwrite(path[:-4] + '.png', (tmpl * 255).astype(np.uint8))
    print(f'template of {len(caps)} caps -> {path} (and .png, the long '
          'tab to the right)')
    global _template
    _template = None
    return tmpl


_template = None


def template():
    """The saved template (:func:`build_template`), long tab at 0 deg."""
    global _template
    if _template is None:
        if not os.path.exists(TEMPLATE_PATH):
            raise RuntimeError(f'no template at {TEMPLATE_PATH} -- build it '
                               'first (menu)')
        data = np.load(TEMPLATE_PATH)
        if 'tmpl' not in data.files or float(data['mm_px']) != MATCH_MM_PX:
            raise RuntimeError(f'{TEMPLATE_PATH} is not a template at '
                               f'{MATCH_MM_PX} mm per px -- build it again')
        _template = data['tmpl']
    return _template


def detect(img, tf_w_cam, tube_xy, seg=None, intrinsics=None, tmpl=None,
           z=Z_CAP):
    """The cap of the tube at ``tube_xy`` (world, m) in ``img``, taken with
    the camera at ``tf_w_cam``: ``dict(long_deg, offset_mm, iou, margin,
    why, top, mask, fit)``. ``long_deg``: the long tab's direction in the
    world (deg, 0 = +x, 90 = +y), None when not trusted (``why`` says);
    ``offset_mm``: the body centre minus ``tube_xy`` (x, y), None without
    a cap; ``iou``: how well the template fits (also when too little for
    a cap); ``margin``: by how much that beats the best fit with the long
    tab the other way round. ``z``: the cap top's height (world, m)."""
    tmpl = template() if tmpl is None else tmpl
    top = top_view(img, tf_w_cam, tube_xy, z=z, intrinsics=intrinsics)
    res = dict(long_deg=None, offset_mm=None, iou=None, margin=None,
               why='no cap', top=top, mask=None, fit=None)
    pieces = candidates(top, seg)
    k, iou, centre = best_fit([_small(p) for p in pieces], tmpl)
    if k is None:
        return res
    if iou.max() < MIN_IOU:
        res.update(iou=float(iou.max()))
        return res
    mask = pieces[k]
    deg = int(np.argmax(iou))
    other = iou[(deg + 180 + np.arange(-45, 46)) % 360].max()
    u, v = centre[deg]
    half = VIEW_HALF * 1000
    res.update(mask=mask, iou=float(iou[deg]),
               margin=float(iou[deg] - other), fit=(deg, (u, v)),
               offset_mm=((u + 0.5) * MATCH_MM_PX - half,
                          half - (v + 0.5) * MATCH_MM_PX), why='')
    if res['margin'] < MIN_MARGIN:
        res['why'] = f'long tab / strap unsure (margin {res["margin"]:.2f})'
    else:
        res['long_deg'] = float(deg)
    return res


def draw(res, tmpl=None):
    """The top view with the cap's mask (blue) and the fitted template
    (green) outlined, and the long tab's direction."""
    shown = res['top'].copy()
    if res['mask'] is None:
        cv2.putText(shown, 'no cap', (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        return shown
    tmpl = template() if tmpl is None else tmpl
    cs, _ = cv2.findContours(res['mask'].astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(shown, cs, -1, (255, 128, 0), 1)
    deg, (u, v) = res['fit']
    k = MATCH_MM_PX / MM_PX                 # small px -> top-view px
    h = tmpl.shape[0]
    ts, _ = cv2.findContours((_turn(tmpl, deg) > 0.5).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ts = [np.round((t - (h - 1) / 2 + (u + 0.5, v + 0.5)) * k)
          .astype(np.int32) for t in ts]
    cv2.drawContours(shown, ts, -1, (0, 200, 0), 1)
    c = (int(round((u + 0.5) * k)), int(round((v + 0.5) * k)))
    t = np.radians(deg)
    reach = 12 / MM_PX                      # 12 mm
    tip = (int(round(c[0] + reach * np.cos(t))),
           int(round(c[1] - reach * np.sin(t))))
    sure = res['long_deg'] is not None
    cv2.arrowedLine(shown, c, tip, (0, 0, 255) if sure else (0, 200, 255),
                    2, tipLength=0.2)
    text = (f'long tab {res["long_deg"]:.0f} deg' if sure
            else f'long tab: ? ({res["why"]})')
    cv2.putText(shown, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    return shown


def report(res):
    if res['offset_mm'] is None:
        return 'no cap'
    long = (f'? ({res["why"]})' if res['long_deg'] is None
            else f'{res["long_deg"]:.0f}')
    dx, dy = res['offset_mm']
    return (f'long tab {long} deg | centre {dx:+.1f} {dy:+.1f} mm | iou '
            f'{res["iou"]:.2f}, margin {res["margin"]:.2f}')


# ---- pictures --------------------------------------------------------------
def grab(n=N_AVG, warm=10):
    """The picker camera's picture now, ``n`` frames averaged."""
    from one.vision.calibrate.cameras import index_of
    cap = cv2.VideoCapture(index_of(calib.PORT), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f'{calib.LABEL}: cannot open')
    try:
        for _ in range(warm):
            cap.read()
        frames = []
        while len(frames) < n:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'{calib.LABEL}: gives no image')
            frames.append(frame.astype(np.float32))
    finally:
        cap.release()
    return np.mean(frames, 0).astype(np.uint8)


def save_sample(img, qs, tube_xy, name, sample_dir=SAMPLE_DIR):
    """``<sample_dir>/<name>.png`` and ``.json`` (joints, tube xy)."""
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(sample_dir, name)
    cv2.imwrite(path + '.png', img)
    with open(path + '.json', 'w', encoding='utf-8') as f:
        json.dump({'qs': [float(q) for q in qs],
                   'tube_xy': [float(v) for v in tube_xy[:2]]}, f)
    return path + '.png'


def scan(at_tube, start=0):
    """On the real picker: tube by tube from ``start`` (the order of
    ``config/poses_pick.json``) to its pre_pick, a picture, then
    ``at_tube(tube, img, tf_w_cam, qs, tube_xy)`` -- False ends the scan.
    Ends too at the last tube, on q / Esc, or when a move cannot be
    planned; the arm is left where it stands, the gripper back at its
    working open. False on a failure."""
    from one.drivers.cobotta import BCapError
    from one.utils.keys import _quit_pressed
    import coripps.picker_motion as pkm
    import coripps.tool.tube.layout_pick as layout
    import coripps.tool.tube.poses_io_pick as ppio
    from coripps.robots.mobisys import mobisys
    from coripps.end_effectors.xh430_gripper.exec import (XH430GripperExec,
                                                          DynamixelError)

    _, picker, sampler = mobisys()
    pkm.freeze_sampler(sampler)
    pln_ctx, _ = pkm.build_planning_context(picker, sampler)
    points, look_qs, _ = ppio.load_checked(ndof=picker.ndof, layout=layout)
    try:
        gripper = XH430GripperExec().connect()
    except DynamixelError as exc:
        print(f'gripper: {exc}')
        return False
    cobotta = pkm.connect(pkm._PICKER_HOST, 'picker', motion=True)
    if cobotta is None:
        gripper.disconnect()
        return False
    gripper.open_for_camera()           # the fingers out of the pictures
    print(f'tubes {start}..{len(points) - 1}; q / Esc stops')
    ok = True
    try:
        for tube in range(start, len(points)):
            if _quit_pressed() or cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                print('quit')
                break
            print(f'\ntube {tube}')
            here = pln_ctx.enforce_bounds(
                cobotta.joint_angles(ndof=picker.ndof))
            if not pln_ctx.is_state_valid(here):
                print('  the current pose is in collision -- stopping')
                ok = False
                break
            way = pkm.build_transfer(pln_ctx, here, look_qs[tube],
                                     force_plan=False)
            if way is None:
                print('  no path to its pre_pick -- stopping')
                ok = False
                break
            cobotta.enable(speed=LIVE_SPEED)
            cobotta.move_path(way)
            qs = cobotta.joint_angles(ndof=picker.ndof)
            img = grab()
            if not at_tube(tube, img, camera_tf(picker, qs), qs,
                           points[tube][:2]):
                break
        else:
            print('\nlast tube done')
    except (OSError, BCapError, RuntimeError, DynamixelError) as exc:
        print(f'  failed: {exc} -- stopping')
        ok = False
    finally:
        cobotta.disconnect()            # motors off, the arm given back
        try:
            gripper.open()              # back to the working open
        except DynamixelError:
            pass
        gripper.disconnect()
    return ok


def _show(shown):
    cv2.imshow(WINDOW, shown)
    cv2.waitKey(1)


def _wait_key():
    """A key in the window (its low byte); Esc (27) when the window is
    closed, or was never opened."""
    while True:
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return 27
        key = cv2.waitKey(100)
        if key >= 0:
            return key & 0xFF


def _close_window():
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
        print('any key in the window (or closing it) ends')
        _wait_key()
    cv2.destroyAllWindows()


def clear_samples(prefix):
    """Delete the samples ``<prefix>*`` (.png and .json)."""
    gone = [p for p in glob.glob(os.path.join(SAMPLE_DIR, prefix + '*'))
            if p.endswith(('.png', '.json'))]
    for p in gone:
        os.remove(p)
    if gone:
        print(f'  deleted the last scan: {len(gone) // 2} {prefix}* samples')


def run_live(start=0, prefix='live_tube'):
    """The scan with the template: each tube's cap and long tab, until a
    tube has no cap (the rack is empty from there). Each picture is saved
    as a sample ``<prefix><tube>``; the last scan's are deleted at the
    first picture."""
    seg = segmenter()
    template()                          # none: stop before anything moves

    def at_tube(tube, img, tf_w_cam, qs, tube_xy):
        if tube == start:                # the first picture
            clear_samples(prefix)
        res = detect(img, tf_w_cam, tube_xy, seg=seg)
        save_sample(img, qs, tube_xy, f'{prefix}{tube:03d}')
        print(f'  {report(res)}')
        _show(draw(res))
        if res['mask'] is None:
            print(f'tube {tube}: no cap -- done')
            return False
        return True

    ok = scan(at_tube, start)
    _close_window()
    return ok


def run_template(start=0, prefix='template_tube'):
    """The scan for the template, no template needed: each tube's cap where
    SAM 3's text and point prompts agree on it (:func:`clean_mask`), until
    a tube has no cap; then :func:`build_template` from those. Each picture
    is saved as a sample ``<prefix><tube>``; the last scan's are deleted at
    the first picture (so nothing is lost when the arm never gets there)."""
    from one.vision.sam3 import Segmenter
    text, point = Segmenter(), segmenter()
    masks, names = [], []

    def at_tube(tube, img, tf_w_cam, qs, tube_xy):
        if tube == start:                # the first picture
            clear_samples(prefix)
        name = f'{prefix}{tube:03d}'
        save_sample(img, qs, tube_xy, name)
        top = top_view(img, tf_w_cam, tube_xy)
        mask, why = clean_mask(top, text, point)
        shown = top.copy()
        if mask is not None:
            masks.append(mask)
            names.append(name)
            cs, _ = cv2.findContours(mask.astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(shown, cs, -1, (0, 200, 0), 1)
        verdict = 'clean' if mask is not None else 'left out'
        print(f'  {verdict} ({why})')
        cv2.putText(shown, verdict, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        _show(shown)
        if why == 'no cap':
            print(f'tube {tube}: no cap -- done')
            return False
        return True

    ok = scan(at_tube, start)
    if len(masks) < MIN_TEMPLATE_CAPS:
        print(f'{len(masks)} clean caps, fewer than {MIN_TEMPLATE_CAPS} -- '
              'no template')
        ok = False
    else:
        build_template(masks, names)
    _close_window()
    return ok


def run_samples():
    from coripps.robots.mobisys import mobisys
    picker = mobisys()[calib._ROBOT]
    paths = sorted(p for p in glob.glob(os.path.join(SAMPLE_DIR, '*.png'))
                   if os.path.exists(p[:-4] + '.json'))
    if not paths:
        print(f'no samples with a .json in {SAMPLE_DIR}')
        return False
    for path in paths:
        with open(path[:-4] + '.json', encoding='utf-8') as f:
            meta = json.load(f)
        res = detect(cv2.imread(path), camera_tf(picker, meta['qs']),
                     meta['tube_xy'])
        print(os.path.basename(path), '|', report(res))
        cv2.imshow(WINDOW, draw(res))
        if _wait_key() in (ord('q'), 27):   # any other key: next
            break
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    from one.vision.calibrate import menu
    sys.exit(0 if menu.run('cap orientation:', [
        ('live: tube by tube to pre_pick, until no cap', run_live),
        ('saved samples', run_samples),
        ('template: tube by tube, the clean caps, build', run_template),
    ]) else 1)
