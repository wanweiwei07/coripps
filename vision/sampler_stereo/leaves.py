"""Leaves from one stereo frame (the rectified left picture + its depth,
``depth.rgbd``): a coordinate frame per leaf and the leaf to cut.

The webcams have no IR-cut filter: the picture is lit by room light, by
the 940 nm light, or both, taken at one fixed exposure
(``camera.StereoCam``). Should it be dark, :func:`normalize` lifts it --
a gamma picked from the picture's brightness, then CLAHE on the
lightness -- and everything below works on that picture.

Per leaf, in the normalized picture (from above):
  - the plant and its leaves: SAM 3 (text prompts 'plant' and 'leaf',
    ``one.vision.sam3``); the plant nearest the image centre, the leaves
    mostly inside it -- when no 'plant' is found (small in the picture),
    the leaves grown by PLANT_GROW_PX into blobs, one per plant
  - blade: each leaf mask opened with a disc of BLADE_OPEN_PX -- its thin
    petiole drops out
  - leaf centre C: the blade pixel farthest from the blade's edge
  - junction J (petiole meets blade): the blade's outline point nearest the
    rosette centre, where the blades' long axes meet
In 3D, from the blade's own depth (``depth_of`` -- e.g.
``depth.leaf_depth``; the blade's edge left out, edge pixels mix leaf and
background):
  - C and J: on their camera rays at the median depth of the blade's depth
    pixels within LOCAL_PX of them, widened up to MAX_LOCAL_PX over holes;
    none there: the blade's median depth (said so); no depth on the blade
    at all: left out (said why)
  - frame: origin C, level -- y from J to C in the horizontal, x = up x y,
    z straight down (the nozzle's way in, upright). The leaf's own pitch is
    not used: C and J ~1 cm apart, the stereo's mm put it anywhere in
    +-30 deg, while a leaf here is about level (J at most ~3 deg over C, a
    small one ~0)
The leaf to cut: the most free room on both sides of the petiole at J (other
blades, and the other petioles taken as lines from their J to the rosette
centre, are in the way), so the blades of the cutter can get in.
"""

import cv2
import numpy as np

from one.vision.sam3 import Segmenter  # noqa: F401 (used as lv.Segmenter)


IN_PLANT = 0.5                  # share of a leaf mask inside the plant mask
DUPLICATE = 0.6                 # a leaf mask this much under smaller ones:
                                # SAM 3's second answer for them (two leaves
                                # as one), dropped
PLANT_GROW_PX = 15              # no 'plant' mask: leaves this near, one plant
BLADE_OPEN_PX = 7               # disc diameter that removes the petioles
MIN_BLADE_PX = 40               # smaller blobs are not blades
BLADE_EDGE_PX = 2               # blade edge left out of the depth
LOCAL_PX = 5                    # depth taken within this of C / J
MAX_LOCAL_PX = 15               # ... widened up to this over holes
MIN_LOCAL_PTS = 5               # ... at least these pixels with depth
# normalize: a picture with a mean (0-255) under DARK_MEAN is lifted, its
# mean brightness to TARGET_MEAN (of the bright end, the WHITE_PCT
# percentile), the gamma kept within GAMMA_MIN..1
DARK_MEAN = 60.0
WHITE_PCT = 99.5
TARGET_MEAN = 0.4
GAMMA_MIN = 0.3
CLAHE_CLIP, CLAHE_TILES = 3.0, (8, 8)
# the room at J for the cutter's blades
CUT_BAND_MM = 5.0               # strip along the petiole, from J back
CLEARANCE_MAX_MM = 20.0         # more room than this counts the same
PETIOLE_PX = 3                  # drawn width of the other leaves' petioles


def tf(pos, rotmat):
    out = np.eye(4)
    out[:3, :3] = rotmat
    out[:3, 3] = np.ravel(pos)
    return out


def normalize(image, force=False):
    """``image`` (BGR) brightened for the segmenter when it is dark (its
    mean under DARK_MEAN) or ``force``: gamma, then CLAHE on the
    lightness; as it is otherwise (both hurt SAM 3 on a well-exposed
    picture)."""
    if not force and image.mean() >= DARK_MEAN:
        return image
    img = image.astype(np.float32)
    white = max(float(np.percentile(img, WHITE_PCT)), 1.0)
    img = np.clip(img / white, 0.0, 1.0)
    mean = float(img.mean())
    gamma = (1.0 if mean >= TARGET_MEAN or mean <= 0
             else max(np.log(TARGET_MEAN) / np.log(mean), GAMMA_MIN))
    out = (255 * img ** gamma).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(CLAHE_CLIP, CLAHE_TILES).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def distinct(masks):
    """``masks`` (bool) without those mostly (DUPLICATE) covered by smaller
    ones kept before them -- SAM 3 at times gives two leaves again as one
    mask, besides each on its own."""
    kept, covered = [], None
    for m in sorted(masks, key=lambda m: m.sum()):
        if covered is not None and (m & covered).sum() > DUPLICATE * m.sum():
            continue
        kept.append(m)
        covered = m.copy() if covered is None else covered | m
    return kept


def plant_mask(image, segmenter, near=None):
    """``(the plant's mask, its leaves' masks)``, or None when no plant is
    seen. The plant: the one nearest pixel ``near`` (default: the
    picture's centre)."""
    all_leaves = distinct(segmenter.masks(image, 'leaf'))
    if not all_leaves:
        return None
    plants = list(segmenter.masks(image, 'plant'))
    if not plants:
        # a small plant is often not seen as one: its leaves grown into
        # each other instead, one blob per plant
        grown = cv2.dilate(np.any(all_leaves, axis=0).astype(np.uint8),
                           np.ones((PLANT_GROW_PX,) * 2, np.uint8))
        n, labels = cv2.connectedComponents(grown)
        plants = [labels == k for k in range(1, n)]
    # the plant nearest ``near``
    h, w = image.shape[:2]
    centre = np.array([w / 2, h / 2]) if near is None else np.asarray(near)
    plant = min(plants, key=lambda m: np.linalg.norm(
        np.argwhere(m)[:, ::-1].mean(0) - centre))
    leaf_masks = [m for m in all_leaves
                  if (m & plant).sum() >= IN_PLANT * m.sum()]
    if not leaf_masks:
        return None
    return plant, leaf_masks


def detect_2d(image, segmenter, near=None, leaf_masks=None):
    """``(rosette centre px, [(blade mask, C px, J px), ...], all leaf
    pixels of the plant)``, or None when no plant is seen. The plant: as
    :func:`plant_mask`'s -- or, ``leaf_masks`` given (one SAM 3 call of
    the caller's, e.g. a track's), those leaves, no plant looked for."""
    if leaf_masks is None:
        found = plant_mask(image, segmenter, near)
        if found is None:
            return None
        leaf_masks = found[1]
    leaf_masks = distinct(leaf_masks)
    if not leaf_masks:
        return None
    leaves = np.any(leaf_masks, axis=0)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLADE_OPEN_PX,) * 2)
    found = []
    for leaf in leaf_masks:
        blade = cv2.morphologyEx(leaf.astype(np.uint8), cv2.MORPH_OPEN, disc)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(blade)
        if n < 2:
            continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[biggest, cv2.CC_STAT_AREA] < MIN_BLADE_PX:
            continue
        mask = (labels == biggest).astype(np.uint8)
        depth = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(np.argmax(depth), depth.shape)
        found.append((mask, np.array([cx, cy], float)))
    if not found:
        return None
    # the rosette centre: the point nearest to every blade's long axis
    normals, offsets = [], []
    for mask, c in found:
        ys, xs = np.nonzero(mask)
        pts = np.column_stack([xs, ys]).astype(float)
        pts -= pts.mean(0)
        axis = np.linalg.eigh(pts.T @ pts)[1][:, 1]
        normal = np.array([-axis[1], axis[0]])
        normals.append(normal)
        offsets.append(normal @ c)
    rosette = np.linalg.lstsq(np.array(normals), np.array(offsets),
                              rcond=None)[0]
    result = []
    for mask, c in found:
        outline = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
        j = outline[np.argmin(np.linalg.norm(outline - rosette, axis=1))]
        result.append((mask, c, j.astype(float)))
    return rosette, result, leaves


def petiole_lines(shape, blades, rosette, skip):
    """A mask of the petioles of every blade but ``skip``, taken as the
    straight line from the blade's J to the rosette centre (the real ones
    are too thin to segment)."""
    lines = np.zeros(shape, np.uint8)
    end = tuple(int(t) for t in np.round(rosette))
    for k, (_, _, j_px) in enumerate(blades):
        if k != skip:
            cv2.line(lines, tuple(int(t) for t in np.round(j_px)), end, 1,
                     PETIOLE_PX)
    return lines > 0


def side_clearance(leaves, petioles, blade, c_px, j_px, mm_per_px):
    """Free room (mm) on either side of the petiole at J, in the image: the
    distance from J, across the leaf axis, to the first pixel of another
    leaf or of the other leaves' ``petioles`` within a CUT_BAND_MM strip on
    the petiole side of J. Capped at CLEARANCE_MAX_MM. ``((left, right),
    (left end px, right end px))``."""
    axis = (c_px - j_px) / np.linalg.norm(c_px - j_px)
    across = np.array([-axis[1], axis[0]])
    h, w = leaves.shape
    # obstacles: plant pixels other than this blade (grown a little) and
    # its own petiole (the line from J away from C)
    own = cv2.dilate(blade, np.ones((5, 5), np.uint8)) > 0
    band = max(int(round(CUT_BAND_MM / mm_per_px)), 1)
    for t in range(band + 1):
        u, v = np.round(j_px - t * axis).astype(int)
        own[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2] = True
    obstacle = (leaves & ~own) | petioles
    max_px = int(round(CLEARANCE_MAX_MM / mm_per_px))
    room, ends = [], []
    for side in (1.0, -1.0):
        free = max_px
        for s in range(2, max_px + 1):
            hit = False
            for t in range(band + 1):
                u, v = np.round(j_px + side * s * across - t * axis).astype(int)
                if 0 <= u < w and 0 <= v < h and obstacle[v, u]:
                    hit = True
                    break
            if hit:
                free = s
                break
        room.append(free * mm_per_px)
        ends.append(j_px + side * free * across)
    return tuple(room), tuple(ends)


def ray(px, tf_w_cam, K):
    """``(origin, direction)`` in the world of the camera ray through
    ``px`` (pinhole, the rectified left frame)."""
    d = np.array([(px[0] - K[0, 2]) / K[0, 0], (px[1] - K[1, 2]) / K[1, 1],
                  1.0])
    return tf_w_cam[:3, 3], tf_w_cam[:3, :3] @ d


def depth_points(mask, depth, K, tf_w_cam):
    """World points of the ``mask`` pixels that have depth."""
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u].astype(np.float64)
    pc = np.column_stack([(u - K[0, 2]) / K[0, 0] * z,
                          (v - K[1, 2]) / K[1, 1] * z, z])
    return pc @ tf_w_cam[:3, :3].T + tf_w_cam[:3, 3]


def point_at(px, inner, depth, K, tf_w_cam):
    """World point of pixel ``px`` at the median depth of the ``inner``
    (blade) pixels with depth within LOCAL_PX of it, the radius widened up
    to MAX_LOCAL_PX over holes; None when there are too few."""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    near_px = (uu - px[0]) ** 2 + (vv - px[1]) ** 2
    valid = inner & (depth > 0)
    for radius in range(LOCAL_PX, MAX_LOCAL_PX + 1, 2):
        zs = depth[valid & (near_px <= radius ** 2)]
        if len(zs) >= MIN_LOCAL_PTS:
            return at_depth(px, float(np.median(zs)), K, tf_w_cam)
    return None


def at_depth(px, z, K, tf_w_cam):
    """World point of pixel ``px`` at depth ``z``."""
    pc = np.array([(px[0] - K[0, 2]) / K[0, 0] * z,
                   (px[1] - K[1, 2]) / K[1, 1] * z, z])
    return tf_w_cam[:3, :3] @ pc + tf_w_cam[:3, 3]


def leaf_frames(color, depth, K, tf_w_cam, segmenter, near=None,
                depth_of=None, leaf_masks=None):
    """``(leaves, rosette px)``, None when no plant is seen. ``leaves``: a
    dict per leaf -- 'C', 'J' (world, m), 'tf' (4x4 frame), 'C_px', 'J_px',
    'blade_pts' (the blade's depth points), 'mask' (its blade, bool),
    'room' (free mm left / right of the petiole at J), 'room_ends' (px).
    ``color``: as
    taken -- it is :func:`normalize`-d here; ``near``: as
    :func:`detect_2d`'s; ``depth_of(mask)``: a blade's own depth (None: no
    match), default ``depth``; ``leaf_masks``: as :func:`detect_2d`'s."""
    detected = detect_2d(normalize(color), segmenter, near, leaf_masks)
    if detected is None:
        return None
    rosette, blades, all_leaves = detected
    erode = np.ones((2 * BLADE_EDGE_PX + 1,) * 2, np.uint8)
    up = np.array([0.0, 0.0, 1.0])
    leaves = []
    for k, (mask, c_px, j_px) in enumerate(blades):
        inner = cv2.erode(mask, erode) > 0
        z_map = depth if depth_of is None else depth_of(mask > 0)
        if z_map is None:
            print(f'leaf {k}: not matched in the right picture -- skipped')
            continue
        zs = z_map[inner & (z_map > 0)]
        if len(zs) < MIN_LOCAL_PTS:
            print(f'leaf {k}: {len(zs)} depth pixels -- skipped')
            continue
        c = point_at(c_px, inner, z_map, K, tf_w_cam)
        j = point_at(j_px, inner, z_map, K, tf_w_cam)
        if c is None or j is None:
            z = float(np.median(zs))
            print(f'leaf {k}: no depth near its {"C" if c is None else "J"} '
                  '-- there the leaf\'s median depth')
            c = at_depth(c_px, z, K, tf_w_cam) if c is None else c
            j = at_depth(j_px, z, K, tf_w_cam) if j is None else j
        y = np.array([c[0] - j[0], c[1] - j[1], 0.0])   # level: J -> C
        y /= np.linalg.norm(y)
        x = np.cross(up, y)
        z = np.cross(x, y)              # straight down: the nozzle's way in
        frame = tf(c, np.column_stack([x, y, z]))
        mm_per_px = np.linalg.norm(j - tf_w_cam[:3, 3]) * 1000 / K[0, 0]
        petioles = petiole_lines(all_leaves.shape, blades, rosette, k)
        room, room_ends = side_clearance(all_leaves, petioles, mask, c_px,
                                         j_px, mm_per_px)
        leaves.append({'C': c, 'J': j, 'tf': frame, 'C_px': c_px,
                       'J_px': j_px, 'mask': mask > 0,
                       'blade_pts': depth_points(inner, z_map, K,
                                                 tf_w_cam),
                       'room': room, 'room_ends': room_ends})
    return leaves, rosette


def choose(leaves):
    """Index of the leaf with the most room for the cutter's blades: the
    largest smaller-side room at J."""
    return int(np.argmax([min(leaf['room']) for leaf in leaves]))


def draw(image, leaves, rosette, chosen):
    """``image`` (e.g. ``normalize(color)``) with C (red), J (blue), the
    leaf axis (yellow), the measured room (green) and the leaf numbers (the
    chosen one starred)."""
    out = image.copy()
    for k, leaf in enumerate(leaves):
        c = tuple(int(t) for t in np.round(leaf['C_px']))
        j = tuple(int(t) for t in np.round(leaf['J_px']))
        for end in leaf['room_ends']:
            cv2.line(out, j, tuple(int(t) for t in np.round(end)),
                     (0, 255, 0), 1)
        cv2.line(out, j, c, (0, 255, 255), 1)
        cv2.circle(out, c, 2, (0, 0, 255), -1)
        cv2.circle(out, j, 2, (255, 0, 0), -1)
        cv2.putText(out, f'{k}*' if k == chosen else str(k),
                    (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    cv2.circle(out, tuple(int(t) for t in np.round(rosette)), 2,
               (255, 255, 255), -1)
    return out
