"""Leaves from one D405 frame (color + depth, both in the depth frame): a
coordinate frame per leaf and the leaf to cut.

Per leaf, in the color image (from above):
  - the plant and its leaves: SAM 3 (text prompts 'plant' and 'leaf', see
    :class:`Segmenter`); the plant nearest the image centre, the leaves
    mostly inside it
  - blade: each leaf mask opened with a disc of BLADE_OPEN_PX -- its thin
    petiole drops out
  - leaf centre C: the blade pixel farthest from the blade's edge
  - junction J (petiole meets blade): the blade's outline point nearest the
    rosette centre, where the blades' long axes meet
In 3D (no plane is fitted to the blade -- a real leaf is not flat):
  - C and J: on their camera rays at the median depth of the blade's depth
    pixels within LOCAL_PX of them (the blade's edge left out -- edge
    pixels mix leaf and background depth; widened up to MAX_LOCAL_PX when
    there are holes)
  - frame: origin C, y from J to C, x horizontal (x = up x y), z = x x y
    (from the leaf's face to its back: down, the plant is seen from above)
The leaf to cut: the most free room on both sides of the petiole at J (other
blades, and the other petioles taken as lines from their J to the rosette
centre, are in the way), so the blades of the cutter can get in.
"""

import cv2
import numpy as np


SAM3_MODEL = 'facebook/sam3'    # gated: needs an HF login with access
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
IN_PLANT = 0.5                  # share of a leaf mask inside the plant mask
BLADE_OPEN_PX = 7               # disc diameter that removes the petioles
MIN_BLADE_PX = 40               # smaller blobs are not blades
BLADE_EDGE_PX = 2               # blade edge left out of the depth
LOCAL_PX = 5                    # depth taken within this of C / J
MAX_LOCAL_PX = 15               # ... widened up to this over holes
MIN_LOCAL_PTS = 5
# the room at J for the cutter's blades
CUT_BAND_MM = 5.0               # strip along the petiole, from J back
CLEARANCE_MAX_MM = 20.0         # more room than this counts the same
PETIOLE_PX = 3                  # drawn width of the other leaves' petioles


def tf(pos, rotmat):
    out = np.eye(4)
    out[:3, :3] = rotmat
    out[:3, 3] = np.ravel(pos)
    return out


class Segmenter:
    """SAM 3 (text-prompted instance segmentation), loaded once -- that
    takes ~45 s; a prompt then ~0.15 s on the GPU."""

    def __init__(self, model_id=SAM3_MODEL):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    def masks(self, image, prompt):
        """Bool masks (n, h, w) of every instance of ``prompt`` in the BGR
        ``image``, best first."""
        from PIL import Image
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=rgb, text=prompt,
                                return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=SCORE_THRESHOLD, mask_threshold=MASK_THRESHOLD,
            target_sizes=inputs.get('original_sizes').tolist())[0]
        masks = result['masks'].cpu().numpy().astype(bool)
        order = np.argsort(-result['scores'].cpu().numpy())
        return masks[order]


def detect_2d(image, segmenter):
    """``(rosette centre px, [(blade mask, C px, J px), ...], all leaf
    pixels of the plant)``, or None when no plant is seen."""
    plants = segmenter.masks(image, 'plant')
    if not len(plants):
        return None
    # the plant nearest the image centre
    h, w = image.shape[:2]
    centre = np.array([w / 2, h / 2])
    plant = min(plants, key=lambda m: np.linalg.norm(
        np.argwhere(m)[:, ::-1].mean(0) - centre))
    leaf_masks = [m for m in segmenter.masks(image, 'leaf')
                  if (m & plant).sum() >= IN_PLANT * m.sum()]
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
    ``px`` (pinhole, the depth frame)."""
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
            z = float(np.median(zs))
            pc = np.array([(px[0] - K[0, 2]) / K[0, 0] * z,
                           (px[1] - K[1, 2]) / K[1, 1] * z, z])
            return tf_w_cam[:3, :3] @ pc + tf_w_cam[:3, 3]
    return None


def leaf_frames(color, depth, K, tf_w_cam, segmenter):
    """``(leaves, rosette px)``, None when no plant is seen. ``leaves``: a
    dict per leaf -- 'C', 'J' (world, m), 'tf' (4x4 frame), 'C_px', 'J_px',
    'blade_pts' (the blade's depth points), 'room' (free mm left / right of
    the petiole at J), 'room_ends' (px)."""
    detected = detect_2d(color, segmenter)
    if detected is None:
        return None
    rosette, blades, all_leaves = detected
    erode = np.ones((2 * BLADE_EDGE_PX + 1,) * 2, np.uint8)
    up = np.array([0.0, 0.0, 1.0])
    leaves = []
    for k, (mask, c_px, j_px) in enumerate(blades):
        inner = cv2.erode(mask, erode) > 0
        c = point_at(c_px, inner, depth, K, tf_w_cam)
        j = point_at(j_px, inner, depth, K, tf_w_cam)
        if c is None or j is None:
            print(f'leaf {k}: no depth near its C or J -- skipped')
            continue
        y = (c - j) / np.linalg.norm(c - j)
        x = np.cross(up, y)             # horizontal
        x /= np.linalg.norm(x)
        z = np.cross(x, y)              # face -> back (down)
        frame = tf(c, np.column_stack([x, y, z]))
        mm_per_px = np.linalg.norm(j - tf_w_cam[:3, 3]) * 1000 / K[0, 0]
        petioles = petiole_lines(all_leaves.shape, blades, rosette, k)
        room, room_ends = side_clearance(all_leaves, petioles, mask, c_px,
                                         j_px, mm_per_px)
        leaves.append({'C': c, 'J': j, 'tf': frame, 'C_px': c_px,
                       'J_px': j_px,
                       'blade_pts': depth_points(inner, depth, K, tf_w_cam),
                       'room': room, 'room_ends': room_ends})
    return leaves, rosette


def choose(leaves):
    """Index of the leaf with the most room for the cutter's blades: the
    largest smaller-side room at J."""
    return int(np.argmax([min(leaf['room']) for leaf in leaves]))


def draw(image, leaves, rosette, chosen):
    """The color image with C (red), J (blue), the leaf axis (yellow), the
    measured room (green) and the leaf numbers (the chosen one starred)."""
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
