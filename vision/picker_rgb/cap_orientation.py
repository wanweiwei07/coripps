"""Which way the tube cap's long tab points, seen by the picker's webcam.

The cap is a round body with a tab sticking out on two sides, one longer
than the other: the long one is the brim, the short one the hinge. The
long tab (brim) has to point to -y. The camera on the picker
looks down at the cap.

Teach (once, ``teach``): a single cap, nothing touching it. Its outline is
traced (``detect_outline``):

1. the cap is the strongest Hough circle near EXPECT_XY (where the cap is
   when the picker stands over the tube);
2. the outline is the Canny edge pieces that start near that circle, stay
   close to it and are not tiny (the rack's edges run far away, its printed
   labels are small);
3. the outline's radius around the body centre, per degree, is r(theta):
   the body gives ~R all round, each tab a bump -- measured from the circle
   RANSAC-fitted to the outline (the tabs are its outliers), not from the
   Hough circle, whose centre wanders by a few px;
4. the bumps are the runs where r sticks out by more than MIN_BUMP; the
   bigger one (area) is the long tab. Its direction is taken as opposite
   the short tab's: the short tab is narrow and its angle steady, the long
   one wide and lumpy. Only two bumps, about opposite, one clearly bigger,
   count as a cap.

The edge pixels inside that outline (less the round middle, which says
nothing about rotation) are the template, saved with the long tab's angle.

Detect (``detect``): the template is turned through 360 deg and shifted
around the Hough circle; each pose is scored by the mean distance from the
template pixels to the nearest image edge (chamfer). Only the template's
own pixels are scored, so a neighbouring cap touching this one, the rack's
holes and labels are clutter that mostly misses, where they broke the
outline tracing. The best rotation plus the taught angle is the long tab;
it is trusted only when it fits (MAX_SCORE) and every pose more than 30 deg
away fits clearly worse (MIN_MARGIN) -- the flipped cap is the one to beat.

Angles are in the image, degrees, 0 = +u (right), 90 = +v (down).

Live view (t teaches from the current frame, space saves it to samples/,
q or Esc quits); or on saved images:

    py -3.12 coripps/vision/picker_rgb/cap_orientation.py
    py -3.12 coripps/vision/picker_rgb/cap_orientation.py samples/cap_001.png
"""

import os
import sys

import cv2
import numpy as np

from one.vision.calibrate.cameras import find_by_port


PORT = 3                        # hub port of the picker camera
EXPECT_XY = (300, 266)          # px, where the cap body is expected
MAX_OFFSET = 50                 # px, body farther than this is not the cap
R_RANGE = (34, 41)              # px, body radius (the inner rings are
                                # circles too; keep them out)
OUTLINE_IN = (0.85, 1.5)        # x R, an outline piece must start in here
OUTLINE_OUT = 1.9               # x R, ... and not reach out beyond this
MIN_PIECE = 40                  # px, smaller edge pieces are labels/specks
MIN_BUMP = 0.12                 # x R, a tab sticks out at least this much
MIN_SPAN = 8                    # deg, ... over at least this many degrees
MAX_SKEW = 35                   # deg, the two tabs are 180 +- this apart
MIN_RATIO = 1.5                 # long tab / short tab size, at least
RANSAC_ITERS = 300              # body circle fit to the outline
RANSAC_TOL = 2.5                # px (the outline is a double line)
MATCH_SHIFT = 12                # px, template slid this far around the circle
MATCH_STEP = 2                  # deg, coarse rotation step (then +-2 by 0.5)
DT_CAP = 4.0                    # px, an edge farther than this counts as this
MAX_SCORE = 1.0                 # px, mean chamfer distance, at most
MIN_MARGIN = 0.2                # px, next best pose >30 deg away is worse by
N_TMPL = 600                    # template pixels kept (speed)
N_AVG = 10                      # frames averaged per detection (noise)
WINDOW = 'cap orientation'
_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(_HERE, 'samples')
TEMPLATE_PATH = os.path.join(_HERE, 'cap_template.npz')


def find_body(gray, expect_xy=EXPECT_XY):
    """``(u, v, r)`` of the cap body near ``expect_xy``, or None."""
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=60, param2=18,
                               minRadius=R_RANGE[0], maxRadius=R_RANGE[1])
    if circles is None:
        return None
    # strongest first; the rack's holes are circles too, hence EXPECT_XY
    for u, v, r in circles[0]:
        if np.hypot(u - expect_xy[0], v - expect_xy[1]) <= MAX_OFFSET:
            return float(u), float(v), float(r)
    return None


def outline_points(gray, body):
    """``(n, 2)`` (u, v) edge pixels of the cap's outline, or None."""
    u, v, r = body
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 20, 50)
    _, labels = cv2.connectedComponents(edges, connectivity=8)
    vs, us = np.nonzero(edges)
    rad = np.hypot(us - u, vs - v)
    lab = labels[vs, us]
    # per piece: its nearest and farthest point from the centre, its size
    n = labels.max() + 1
    rmin = np.full(n, np.inf)
    rmax = np.zeros(n)
    np.minimum.at(rmin, lab, rad)
    np.maximum.at(rmax, lab, rad)
    size = np.bincount(lab, minlength=n)
    keep = ((rmin > OUTLINE_IN[0] * r) & (rmin < OUTLINE_IN[1] * r)
            & (rmax < OUTLINE_OUT * r) & (size >= MIN_PIECE))[lab]
    if keep.sum() < 50:
        return None
    return np.stack([us[keep], vs[keep]], 1).astype(np.float64)


def profile(pts, centre):
    """Outline radius per degree around ``centre`` (360,), px, or None."""
    d = pts - centre
    rad = np.hypot(d[:, 0], d[:, 1])
    deg = np.degrees(np.arctan2(d[:, 1], d[:, 0])).astype(int) % 360
    prof = np.full(360, np.nan)
    np.fmax.at(prof, deg, rad)
    # fill gaps (circularly) and smooth a little
    ok = ~np.isnan(prof)
    if ok.sum() < 90:
        return None
    idx = np.arange(360)
    prof = np.interp(idx, idx[ok], prof[ok], period=360)
    k = np.ones(5) / 5
    return np.convolve(np.r_[prof[-2:], prof, prof[:2]], k, 'valid')


def fit_circle(pts):
    """Least-squares ``(u, v, r)`` through ``pts``."""
    a = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    (u, v, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(u), float(v), float(np.sqrt(c + u * u + v * v))


def body_circle(pts, rng=None):
    """``(u, v, r)`` of the round part of the outline: RANSAC over circles
    through 3 outline pixels (radius in R_RANGE), the tabs are outliers;
    then least squares on the inliers. None if too few agree."""
    rng = np.random.default_rng(0) if rng is None else rng
    best, best_n = None, 0
    for _ in range(RANSAC_ITERS):
        trio = pts[rng.choice(len(pts), 3, replace=False)]
        try:
            u, v, r = fit_circle(trio)
        except np.linalg.LinAlgError:
            continue
        if not R_RANGE[0] <= r <= R_RANGE[1]:
            continue
        inl = np.abs(np.hypot(*(pts - (u, v)).T) - r) < RANSAC_TOL
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    if best is None or best_n < 0.25 * len(pts):
        return None
    return fit_circle(pts[best])


def bumps(prof, r):
    """The tabs: ``[(length px, direction deg, (start, end) deg, size),
    ...]``, biggest first. ``size`` is the bump's area in the profile
    (px x deg): steadier than its peak, which one noisy edge pixel moves."""
    base = np.median(prof)
    excess = prof - base
    out = excess > MIN_BUMP * r
    if out.all() or not out.any():
        return []
    # start the walk at an angle that is not in a bump
    start = int(np.argmin(out))
    found = []
    i = 0
    while i < 360:
        a = (start + i) % 360
        if not out[a]:
            i += 1
            continue
        run = []
        while i < 360 and out[(start + i) % 360]:
            run.append((start + i) % 360)
            i += 1
        if len(run) < MIN_SPAN:
            continue
        run = np.array(run)
        w = excess[run]
        t = np.radians(run)
        direction = np.degrees(np.arctan2((w * np.sin(t)).sum(),
                                          (w * np.cos(t)).sum())) % 360
        found.append((float(w.max()), float(direction),
                      (int(run[0]), int(run[-1])), float(w.sum())))
    return sorted(found, key=lambda t: t[3], reverse=True)


def detect_outline(img, expect_xy=EXPECT_XY):
    """``dict(body, profile, tabs, long_deg)``; ``long_deg`` is None
    when the long tab cannot be told.

    The Hough circle only finds the cap: its centre wanders by a few px
    from frame to frame (the cap has concentric rings), about the
    difference between the two tabs. The tabs are measured from the circle
    fitted to the outline itself."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    res = dict(body=None, profile=None, tabs=[], long_deg=None)
    body = find_body(gray, expect_xy)
    if body is None:
        return res
    pts = outline_points(gray, body)
    if pts is None:
        res['body'] = body
        return res
    res['body'] = body
    body = body_circle(pts)
    if body is None:
        return res
    res['body'] = body
    prof = profile(pts, body[:2])
    if prof is None:
        return res
    res['profile'] = prof
    res['tabs'] = tabs = bumps(prof, body[2])
    # trust it only when it looks like the cap: two tabs, about opposite,
    # one clearly longer
    if len(tabs) == 2:
        apart = abs((tabs[0][1] - tabs[1][1] + 180) % 360 - 180)
        if (abs(apart - 180) <= MAX_SKEW
                and tabs[0][3] >= MIN_RATIO * tabs[1][3]):
            # the short tab is narrow, its angle steady (+-2 deg); the long
            # one is wide and lumpy (+-10): it only says which side
            res['long_deg'] = (tabs[1][1] + 180) % 360
    return res


def draw_outline(img, res):
    shown = img.copy()
    if res['body'] is None:
        cv2.putText(shown, 'no cap', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        return shown
    u, v, r = res['body']
    c = (int(round(u)), int(round(v)))
    cv2.circle(shown, c, int(round(r)), (0, 200, 0), 1)
    if res['profile'] is not None:
        t = np.radians(np.arange(360))
        pts = np.stack([u + res['profile'] * np.cos(t),
                        v + res['profile'] * np.sin(t)], 1)
        cv2.polylines(shown, [pts.round().astype(np.int32)], True,
                      (255, 128, 0), 1)
    def arrow(deg, reach, color, width):
        t = np.radians(deg)
        tip = (int(round(u + reach * np.cos(t))),
               int(round(v + reach * np.sin(t))))
        cv2.arrowedLine(shown, c, tip, color, width, tipLength=0.2)

    for length, deg, _, _ in res['tabs']:   # every bump found, thin
        arrow(deg, r + length, (0, 200, 255), 1)
    if res['long_deg'] is None:
        text = 'long tab: ?'
    else:
        arrow(res['long_deg'], 1.8 * r, (0, 0, 255), 2)
        text = (f'long tab {res["long_deg"]:.0f} deg  (size '
                f'{res["tabs"][0][3] / res["tabs"][1][3]:.1f}x)')
    cv2.putText(shown, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    return shown


def report_outline(res):
    if res['body'] is None:
        return 'no cap'
    u, v, r = res['body']
    tabs = ', '.join(f'{d:.0f} deg len {l:.1f}px ({l / r:.2f}R) '
                     f'span {s}-{e} size {a:.0f}'
                     for l, d, (s, e), a in res['tabs'])
    return f'body ({u:.0f}, {v:.0f}) r {r:.1f} | tabs: {tabs or "none"}'


def _edges(gray):
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 20, 50)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def teach(img, expect_xy=EXPECT_XY, path=TEMPLATE_PATH):
    """Template from a frame with one clean cap: ``dict(pts, long_deg)``,
    ``pts`` (n, 2) edge pixels relative to the body centre. Saved to
    ``path`` (None: not saved). None if the cap's outline cannot be told."""
    res = detect_outline(img, expect_xy)
    if res['long_deg'] is None:
        return None
    u, v, r = res['body']
    t = np.radians(np.arange(360))
    poly = np.stack([u + (res['profile'] + 3) * np.cos(t),
                     v + (res['profile'] + 3) * np.sin(t)], 1)
    inside = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(inside, [poly.round().astype(np.int32)], 255)
    vs, us = np.nonzero((_edges(_gray(img)) > 0) & (inside > 0))
    pts = np.stack([us - u, vs - v], 1).astype(np.float64)
    pts = pts[np.hypot(*pts.T) > 0.6 * r]
    if len(pts) > N_TMPL:
        pts = pts[np.random.default_rng(0).choice(len(pts), N_TMPL,
                                                  replace=False)]
    tmpl = dict(pts=pts, long_deg=float(res['long_deg']))
    if path is not None:
        np.savez(path, **tmpl)
    return tmpl


def load_template(path=TEMPLATE_PATH):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return dict(pts=d['pts'], long_deg=float(d['long_deg']))


def _chamfer(dt, pts, degs, xy, shifts):
    """Best mean distance per rotation in ``degs`` over the ``shifts``:
    ``(scores (len(degs),), (dx, dy) per rotation)``."""
    h, w = dt.shape
    a = np.radians(degs)[:, None]
    ru = pts[:, 0] * np.cos(a) - pts[:, 1] * np.sin(a) + xy[0]
    rv = pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a) + xy[1]
    best = np.full(len(degs), np.inf)
    where = np.zeros((len(degs), 2))
    for dx, dy in shifts:
        cu = np.clip(np.round(ru + dx).astype(int), 0, w - 1)
        cv = np.clip(np.round(rv + dy).astype(int), 0, h - 1)
        sc = dt[cv, cu].mean(1)
        better = sc < best
        best[better] = sc[better]
        where[better] = dx, dy
    return best, where


def detect(img, expect_xy=EXPECT_XY, tmpl=None):
    """``dict(centre, rot, long_deg, score, margin, pts)``; ``long_deg``
    None when the match is not trusted, everything None when no cap or no
    template."""
    tmpl = load_template() if tmpl is None else tmpl
    res = dict(centre=None, rot=None, long_deg=None, score=None,
               margin=None, pts=None)
    gray = _gray(img)
    body = find_body(gray, expect_xy)
    if body is None or tmpl is None:
        return res
    dt = cv2.distanceTransform((_edges(gray) == 0).astype(np.uint8),
                               cv2.DIST_L2, 3)
    dt = np.minimum(dt, DT_CAP)
    pts = tmpl['pts']
    # coarse: every MATCH_STEP deg, shifts on a 2 px grid
    grid = np.arange(-MATCH_SHIFT, MATCH_SHIFT + 1, 2)
    degs = np.arange(0, 360, MATCH_STEP, dtype=np.float64)
    scores, where = _chamfer(dt, pts, degs, body[:2],
                             [(dx, dy) for dx in grid for dy in grid])
    k = int(np.argmin(scores))
    far = np.abs((degs - degs[k] + 180) % 360 - 180) > 30
    # fine: around the best, 0.5 deg and 1 px
    xy = (body[0] + where[k][0], body[1] + where[k][1])
    fine = degs[k] + np.arange(-MATCH_STEP, MATCH_STEP + 0.25, 0.5)
    fs, fw = _chamfer(dt, pts, fine, xy,
                      [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    j = int(np.argmin(fs))
    rot = float(fine[j] % 360)
    centre = (xy[0] + fw[j][0], xy[1] + fw[j][1])
    a = np.radians(rot)
    res.update(centre=centre, rot=rot, score=float(fs[j]),
               margin=float(scores[far].min() - fs[j]),
               pts=np.stack([pts[:, 0] * np.cos(a) - pts[:, 1] * np.sin(a),
                             pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a)],
                            1) + centre)
    if res['score'] <= MAX_SCORE and res['margin'] >= MIN_MARGIN:
        res['long_deg'] = (tmpl['long_deg'] + rot) % 360
    return res


def draw(img, res):
    shown = img.copy()
    if res['centre'] is None:
        cv2.putText(shown, 'no cap / no template', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return shown
    h, w = shown.shape[:2]
    p = res['pts'].round().astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    color = (0, 200, 0) if res['long_deg'] is not None else (0, 200, 255)
    shown[p[ok, 1], p[ok, 0]] = color
    if res['long_deg'] is None:
        text = 'long tab: ?'
    else:
        c = np.array(res['centre'])
        t = np.radians(res['long_deg'])
        tip = c + 70 * np.array([np.cos(t), np.sin(t)])
        cv2.arrowedLine(shown, tuple(c.round().astype(int)),
                        tuple(tip.round().astype(int)), (0, 0, 255), 2,
                        tipLength=0.2)
        text = f'long tab {res["long_deg"]:.0f} deg'
    text += f'  (fit {res["score"]:.2f}, margin {res["margin"]:.2f})'
    cv2.putText(shown, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    return shown


def report(res):
    if res['centre'] is None:
        return 'no cap / no template'
    long = '?' if res['long_deg'] is None else f'{res["long_deg"]:.1f}'
    return (f'centre ({res["centre"][0]:.0f}, {res["centre"][1]:.0f}) '
            f'long tab {long} deg | fit {res["score"]:.2f} '
            f'margin {res["margin"]:.2f}')


def live():
    by_port = find_by_port()
    if PORT not in by_port:
        print(f'no camera on hub port {PORT} (found: {by_port})')
        return False
    cap = cv2.VideoCapture(by_port[PORT], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('picker camera: cannot open')
        return False
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    tmpl = load_template()
    if tmpl is None:
        print('no template yet: put one cap alone under the camera, press t')
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('picker camera: lost')
                return False
            frames = (frames + [frame.astype(np.float32)])[-N_AVG:]
            img = np.mean(frames, 0).astype(np.uint8)
            res = detect(img, tmpl=tmpl)
            cv2.imshow(WINDOW, draw(img, res))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                new = teach(img)
                if new is None:
                    print('teach: no clean cap outline here, template kept')
                else:
                    tmpl = new
                    print(f'taught: {len(tmpl["pts"])} px, long tab '
                          f'{tmpl["long_deg"]:.1f} deg -> {TEMPLATE_PATH}')
            elif key == ord(' '):
                n = len([f for f in os.listdir(SAMPLE_DIR)
                         if f.endswith('.png')])
                path = os.path.join(SAMPLE_DIR, f'cap_{n:03d}.png')
                cv2.imwrite(path, img)
                print(path, '|', report(res))
            elif key in (ord('q'), 27):
                return True
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        return live()
    for path in sys.argv[1:]:
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                path)
        img = cv2.imread(path)
        res = detect(img)
        print(os.path.basename(path), '|', report(res))
        cv2.imshow(WINDOW, draw(img, res))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    main()
