"""A coordinate frame on every leaf of the plant, from the survey images
(``capture_survey.py``), shown in 3D with mobisys at its real pose.

Per leaf, in the start image (from above):
  - blades: the plant's leaf pixels (``capture_survey.plant_leaves``)
    opened with a disc of BLADE_OPEN_PX -- the thin petioles drop out, each
    blade is left on its own
  - leaf centre C: the blade pixel farthest from the blade's edge
  - junction J (petiole meets blade): the blade's outline point nearest the
    rosette centre, which is where the blades' long axes meet
In 3D:
  - the blade's plane: RANSAC through the plane-sweep points
    (``reconstruct_survey.sweep``) on the blade -- its outline, mostly, the
    inside has no texture to match
  - C and J: their camera rays cut by that plane
  - frame: origin C, y from J to C in the plane, z the plane normal from
    the leaf's face to its back (down: the plant is seen from above, face
    up), x = y x z. A third point off the C-J line is C + x.

The leaf to cut is the one with the most free room on both sides of its
petiole at J (:func:`side_clearance`), so the blades can get in.

Prints each leaf's frame and room, saves the start image with C, J, the
axis and the measured room drawn to ``survey_images/leaf_frames.png``, then shows in 3D: the
plane-sweep point cloud, each blade's fitted plane (a thin patch over its
points), C and J, the frames, and the robots set to the joint angles read
from the controllers (sim pose when one cannot be reached).

    py -3.12 coripps/vision/sampler_rgb_sgl/leaf_frames.py
    py -3.12 coripps/vision/sampler_rgb_sgl/leaf_frames.py --no-show
"""

import os
import sys
import argparse

import cv2
import numpy as np

import one.geom.fitting as ogf

from coripps.vision.sampler_rgb_sgl import calib
from coripps.vision.sampler_rgb_sgl import capture_survey as cs
from coripps.vision.sampler_rgb_sgl import reconstruct_survey as rs


BLADE_OPEN_PX = 7               # disc diameter that removes the petioles
MIN_BLADE_PX = 40               # smaller blobs are not blades
EDGE_SLACK_PX = 2               # sweep points this far outside a blade count
MIN_PLANE_PTS = 20
RANSAC_ITERS = 300
RANSAC_TOL = 0.0015             # m
FRAME_LENGTH = 0.02             # m, drawn axis length
# leaf choice: free room on both sides of the petiole at J, for the blades
CUT_BAND_MM = 5.0               # strip along the petiole, from J back
CLEARANCE_MAX_MM = 20.0         # more room than this counts the same
PETIOLE_PX = 3                  # drawn width of the other leaves' petioles
POINT_RADIUS = 0.001            # m, drawn C and J
PLANE_ALPHA = 0.4
IMAGE_PATH = os.path.join(cs.DATA_DIR, 'leaf_frames.png')
_SAMPLER_HOST = '192.168.0.101'
_PICKER_HOST = '192.168.0.102'


def detect_2d(image):
    """``(rosette centre px, [(blade mask, C px, J px), ...], all leaf
    pixels)``, or None when no plant is seen."""
    leaves = cs.plant_leaves(image)
    if leaves is None:
        return None
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLADE_OPEN_PX,) * 2)
    blades = cv2.morphologyEx(leaves.astype(np.uint8), cv2.MORPH_OPEN, disc)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blades)
    found = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_BLADE_PX:
            continue
        mask = (labels == i).astype(np.uint8)
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


def ray(px, tf_w_cam, K, D):
    """``(origin, direction)`` in the world of the camera ray through
    ``px``."""
    norm = cv2.fisheye.undistortPoints(
        np.asarray(px, np.float64).reshape(1, 1, 2), K, D).ravel()
    return tf_w_cam[:3, 3], tf_w_cam[:3, :3] @ np.array([norm[0], norm[1],
                                                          1.0])


def petiole_lines(shape, blades, rosette, skip):
    """A mask of the petioles of every blade but ``skip``, taken as the
    straight line from the blade's J to the rosette centre (the real ones
    are too thin and grey to segment)."""
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
    leaf or of the other leaves' ``petioles`` (a mask) within a CUT_BAND_MM
    strip on the petiole side of J.
    Capped at CLEARANCE_MAX_MM. ``((left, right), (left end px, right end
    px))``."""
    axis = (c_px - j_px) / np.linalg.norm(c_px - j_px)
    across = np.array([-axis[1], axis[0]])
    h, w = leaves.shape
    # obstacles: plant pixels other than this blade (grown a little) and
    # its own petiole (the line from J away from C)
    own = cv2.dilate(blade, np.ones((5, 5), np.uint8)) > 0
    band = max(int(round(CUT_BAND_MM / mm_per_px)), 1)
    stem = [np.round(j_px - t * axis).astype(int) for t in range(band + 1)]
    for u, v in stem:
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


def ransac_plane(pts, rng):
    """``(unit normal pointing up, d, inlier mask)`` with n . p + d = 0."""
    best = None
    for _ in range(RANSAC_ITERS):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs((pts - a) @ n) < RANSAC_TOL
        if best is None or inliers.sum() > best.sum():
            best = inliers
    n, d = ogf.fit_plane_from_pts(pts[best])
    n, d = np.asarray(n, np.float64), float(d)
    if n[2] < 0:
        n, d = -n, -d
    inliers = np.abs(pts @ n + d) < RANSAC_TOL
    return n, d, inliers


def leaf_frames(images, cams, K, D):
    """``(leaves, rosette px, (cloud points, cloud colors))``, None when
    no plant is seen. ``leaves``: a dict per leaf -- 'C', 'J' (world, m),
    'tf' (4x4 frame), 'C_px', 'J_px', 'n_pts', 'n_inliers', 'rms' (m),
    'plane_pts' (its inlier points), 'room' (free mm left / right of the
    petiole at J, :func:`side_clearance`), 'room_ends' (px)."""
    detected = detect_2d(images[0])
    if detected is None:
        return None
    rosette, blades, all_leaves = detected
    box = rs.plant_box(images[0])
    pts, colors, _, pixels = rs.sweep(images, cams, K, D, box)
    kernel = np.ones((2 * EDGE_SLACK_PX + 1,) * 2, np.uint8)
    rng = np.random.default_rng(0)
    leaves = []
    for k, (mask, c_px, j_px) in enumerate(blades):
        near = cv2.dilate(mask, kernel)[pixels[:, 1], pixels[:, 0]] > 0
        if near.sum() < MIN_PLANE_PTS:
            print(f'leaf {k}: only {near.sum()} 3D points on it -- skipped')
            continue
        n, d, inliers = ransac_plane(pts[near], rng)
        resid = pts[near][inliers] @ n + d
        on_plane = []
        for px in (c_px, j_px):
            o, v = ray(px, cams[0], K, D)
            on_plane.append(o - (n @ o + d) / (n @ v) * v)
        c, j = on_plane
        z = -n                          # face -> back
        y = (c - j) - ((c - j) @ z) * z
        y /= np.linalg.norm(y)
        tf = np.eye(4)
        tf[:3, :3] = np.column_stack([np.cross(y, z), y, z])
        tf[:3, 3] = c
        mm_per_px = np.linalg.norm(j - cams[0][:3, 3]) * 1000 / K[0, 0]
        petioles = petiole_lines(all_leaves.shape, blades, rosette, k)
        room, room_ends = side_clearance(all_leaves, petioles, mask, c_px,
                                         j_px, mm_per_px)
        leaves.append({'C': c, 'J': j, 'tf': tf, 'C_px': c_px, 'J_px': j_px,
                       'room': room, 'room_ends': room_ends,
                       'n_pts': int(near.sum()),
                       'n_inliers': int(inliers.sum()),
                       'rms': float(np.sqrt(np.mean(resid ** 2))),
                       'plane_pts': pts[near][inliers]})
    return leaves, rosette, (pts, colors)


def choose(leaves):
    """Index of the leaf with the most room for the blades: the largest
    smaller-side room at J."""
    return int(np.argmax([min(leaf['room']) for leaf in leaves]))


def draw(image, leaves, rosette, chosen):
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


def read_qs(host, label, ndof=6):
    """Joint angles from a controller, or None (said why)."""
    from one.drivers.cobotta import BCapError, Cobotta
    try:
        with Cobotta(host) as cobotta:
            qs = cobotta.joint_angles(ndof=ndof)
    except (OSError, BCapError) as exc:
        print(f'{label}: cannot read {host} ({exc}) -- shown at its sim pose')
        return None
    print(f'{label}: {np.round(np.degrees(qs), 2)} deg')
    return qs


def show(leaves, cloud, chosen):
    import one.scene.scene_object_primitive as ossop
    import one.utils.constant as ouc
    import one.viewer.world as ovw
    from coripps.robots.mobisys import mobisys

    center = np.mean([leaf['C'] for leaf in leaves], axis=0)
    base = ovw.World(cam_pos=center + np.array([0.45, 0.35, 0.35]),
                     cam_lookat_pos=center)
    mobisys_base, picker, sampler = mobisys()
    for robot, host, label in ((picker, _PICKER_HOST, 'picker'),
                               (sampler, _SAMPLER_HOST, 'sampler')):
        qs = read_qs(host, label, robot.ndof)
        if qs is not None:
            robot.fk(qs=qs)
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    ossop.point_cloud(cloud[0], cloud[1]).attach_to(base.scene)
    scale = FRAME_LENGTH / 0.2          # the frame primitive is 0.2 m long
    for k, leaf in enumerate(leaves):
        tf = leaf['tf']
        # the plane: a thin patch over the blade's inlier points
        local = (leaf['plane_pts'] - tf[:3, 3]) @ tf[:3, :3]
        lo, hi = local[:, :2].min(0), local[:, :2].max(0)
        mid = tf[:3, :3] @ np.array([*(lo + hi) / 2, 0.0]) + tf[:3, 3]
        ossop.box(pos=mid, xyz_lengths=(*(hi - lo), 2e-4),
                  rotmat=tf[:3, :3], rgb=ouc.BasicColor.YELLOW,
                  alpha=PLANE_ALPHA).attach_to(base.scene)
        ossop.sphere(pos=leaf['C'], radius=POINT_RADIUS,
                     rgb=ouc.BasicColor.RED).attach_to(base.scene)
        ossop.sphere(pos=leaf['J'], radius=POINT_RADIUS,
                     rgb=ouc.BasicColor.BLUE).attach_to(base.scene)
        # the chosen leaf's frame twice as long
        length = scale * (2 if k == chosen else 1)
        ossop.frame_from_tf(tf, length_scale=length,
                            radius_scale=scale * 2).attach_to(base.scene)
    base.run()


def main(show_3d=True):
    images, cams = rs.load()
    K, D = calib.load_intrinsics()
    found = leaf_frames(images, cams, K, D)
    if found is None or not found[0]:
        print('no leaves found')
        return False
    leaves, rosette, cloud = found
    for k, leaf in enumerate(leaves):
        tilt = np.degrees(np.arccos(min(-leaf['tf'][2, 2], 1.0)))
        print(f"leaf {k}: C {np.round(leaf['C'] * 1000, 1)} mm  "
              f"J {np.round(leaf['J'] * 1000, 1)} mm  "
              f"C-J {np.linalg.norm(leaf['C'] - leaf['J']) * 1000:.1f} mm  "
              f"tilt {tilt:.1f} deg  plane {leaf['n_inliers']}/"
              f"{leaf['n_pts']} pts, RMS {leaf['rms'] * 1000:.2f} mm  "
              f"room at J {leaf['room'][0]:.1f} / {leaf['room'][1]:.1f} mm")
    chosen = choose(leaves)
    print(f'chosen: leaf {chosen} ({min(leaves[chosen]["room"]):.1f} mm '
          'on its tighter side)')
    cv2.imwrite(IMAGE_PATH, draw(images[0], leaves, rosette, chosen))
    print(IMAGE_PATH)
    if show_3d:
        show(leaves, cloud, chosen)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--no-show', action='store_true',
                        help='print and save only, no viewer')
    args = parser.parse_args()
    sys.exit(0 if main(show_3d=not args.no_show) else 1)
