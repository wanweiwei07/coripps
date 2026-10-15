"""3D point cloud around the plant from the survey images
(``capture_survey.py``), by a multi-view plane sweep.

The first image (the start view, from above) is the reference, cropped to
the box around the plant's leaves (``capture_survey.plant_leaves``) plus
ROI_MARGIN: the leaves only say where to look, every pixel in the box is
tried -- leaf, soil, rim alike. For each, horizontal planes from Z_BELOW under the table to Z_ABOVE over
it are tried: the pixel's ray is cut by the plane, the point projected into
every other image, and the patch around it compared with the reference
patch (NCC). The height where the images agree best is the pixel's. Pixels
whose patch has no texture, whose best match is weak, or whose best height
is at an end of the sweep (out of range) are left out rather than guessed.

Prints the extent and shows the points.

    py -3.12 coripps/vision/sampler_rgb_sgl/reconstruct_survey.py
    py -3.12 coripps/vision/sampler_rgb_sgl/reconstruct_survey.py --no-show
"""

import os
import sys
import json
import time
import argparse

import cv2
import numpy as np

from coripps.vision.sampler_rgb_sgl import calib
from coripps.vision.sampler_rgb_sgl import capture_survey as cs


Z_BELOW = 0.01                  # m under the table, bottom of the sweep
Z_ABOVE = 0.15                  # m over the table, top of the sweep
STEP = 0.001                    # m between planes (refined by a parabola)
PATCH = 9                       # px, NCC window
MIN_VIEWS = 3                   # other images that must see a point
MIN_STD = 6.0                   # grey levels: flatter patches are skipped
MIN_NCC = 0.6                   # weaker best matches are skipped
ROI_MARGIN = 20                 # px around the leaves' box
PLY_PATH = os.path.join(cs.DATA_DIR, 'points.ply')


def load():
    """``(images, world camera tfs)`` of the survey."""
    with open(cs.RECORD_PATH, encoding='utf-8') as f:
        records = json.load(f)
    images = [cv2.imread(os.path.join(cs.DATA_DIR, r['image']))
              for r in records]
    tf_flange_cam = cs.load_handeye()
    cams = [w @ tf_flange_cam
            for w in calib.flange_tfs([r['qs'] for r in records])]
    return images, cams


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _local_stats(a, b=None):
    """Box-filtered variance of ``a`` (and of ``b`` and the covariance)."""
    k = (PATCH, PATCH)
    mean_a = cv2.blur(a, k)
    var_a = cv2.blur(a * a, k) - mean_a ** 2
    if b is None:
        return var_a
    mean_b = cv2.blur(b, k)
    var_b = cv2.blur(b * b, k) - mean_b ** 2
    cov = cv2.blur(a * b, k) - mean_a * mean_b
    return var_a, var_b, cov


def _plane_score(ref, origin, dirs, z, others, K, D):
    """Mean NCC over the other images of every reference pixel put at
    height ``z``; -1 where fewer than MIN_VIEWS images see it."""
    h, w = ref.shape
    img_h, img_w = others[0][0].shape
    pts = origin + ((z - origin[2]) / dirs[:, 2])[:, None] * dirs
    total = np.zeros(ref.shape, np.float32)
    count = np.zeros(ref.shape, np.float32)
    for gray, tf_cam_w in others:
        pc = pts @ tf_cam_w[:3, :3].T + tf_cam_w[:3, 3]
        px, _ = cv2.fisheye.projectPoints(
            np.ascontiguousarray(pc.reshape(-1, 1, 3)), np.zeros((3, 1)),
            np.zeros((3, 1)), K, D)
        px = px.reshape(h, w, 2).astype(np.float32)
        valid = ((pc[:, 2] > 1e-3).reshape(h, w)
                 & (px[..., 0] >= 0) & (px[..., 0] < img_w - 1)
                 & (px[..., 1] >= 0) & (px[..., 1] < img_h - 1))
        warped = cv2.remap(gray, px[..., 0], px[..., 1], cv2.INTER_LINEAR)
        var_a, var_b, cov = _local_stats(ref, warped)
        ncc = cov / np.sqrt(np.maximum(var_a * var_b, 1e-6))
        ok = valid & (var_b > MIN_STD ** 2)
        total[ok] += ncc[ok]
        count[ok] += 1
    score = np.full(ref.shape, -1.0, np.float32)
    enough = count >= MIN_VIEWS
    score[enough] = total[enough] / count[enough]
    return score


def plant_box(image):
    """``(x0, x1, y0, y1)`` of the plant's leaves plus ROI_MARGIN, or
    None."""
    leaves = cs.plant_leaves(image)
    if leaves is None:
        return None
    ys, xs = np.nonzero(leaves)
    h, w = leaves.shape
    return (max(xs.min() - ROI_MARGIN, 0), min(xs.max() + ROI_MARGIN + 1, w),
            max(ys.min() - ROI_MARGIN, 0), min(ys.max() + ROI_MARGIN + 1, h))


def sweep(images, cams, K, D, box):
    """``(points, colors 0-1, ncc, pixels)`` of the reference pixels in
    ``box`` (x0, x1, y0, y1) that matched; ``pixels`` (n, 2) are their
    (u, v) in the reference image."""
    x0, x1, y0, y1 = box
    ref = _gray(images[0])[y0:y1, x0:x1]
    uu, vv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    norm = cv2.fisheye.undistortPoints(
        np.stack([uu, vv], -1).reshape(-1, 1, 2).astype(np.float64),
        K, D).reshape(-1, 2)
    dirs = np.column_stack([norm, np.ones(len(norm))]) @ cams[0][:3, :3].T
    origin = cams[0][:3, 3]
    others = [(_gray(img), np.linalg.inv(cam))
              for img, cam in zip(images[1:], cams[1:])]
    heights = np.arange(cs.TABLE_Z - Z_BELOW, cs.TABLE_Z + Z_ABOVE, STEP)
    # running best, keeping the scores just before and after it for the
    # sub-step parabola (a full score volume would not fit comfortably)
    best = np.full(ref.shape, -1.0, np.float32)
    best_k = np.zeros(ref.shape, np.int32)
    before = np.full(ref.shape, -1.0, np.float32)
    after = np.full(ref.shape, -1.0, np.float32)
    last = np.full(ref.shape, -1.0, np.float32)
    for k, z in enumerate(heights):
        score = _plane_score(ref, origin, dirs, z, others, K, D)
        just_after = best_k == k - 1
        after[just_after] = score[just_after]
        better = score > best
        before[better] = last[better]
        best[better] = score[better]
        best_k[better] = k
        after[better] = -1.0
        last = score
    keep = ((_local_stats(ref) > MIN_STD ** 2) & (best > MIN_NCC)
            & (best_k > 0) & (best_k < len(heights) - 1))   # not at the ends
    s0, s1, s2 = before[keep], best[keep], after[keep]
    denom = s0 - 2 * s1 + s2
    offset = np.where(np.abs(denom) > 1e-6, 0.5 * (s0 - s2) / denom, 0.0)
    z = heights[best_k[keep]] + np.clip(offset, -1, 1) * STEP
    d = dirs.reshape(ref.shape + (3,))[keep]
    pts = origin + ((z - origin[2]) / d[:, 2])[:, None] * d
    colors = images[0][y0:y1, x0:x1][keep][:, ::-1] / 255.0
    pixels = np.column_stack([uu[keep], vv[keep]])
    return pts, colors, s1, pixels


def write_ply(path, pts, colors):
    rgb = np.round(colors * 255).astype(np.uint8)
    with open(path, 'w', encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\n'
                f'element vertex {len(pts)}\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                'end_header\n')
        for p, c in zip(pts, rgb):
            f.write(f'{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n')


def main(show=True):
    images, cams = load()
    K, D = calib.load_intrinsics()
    box = plant_box(images[0])
    if box is None:
        print('no plant in the start image')
        return False
    t0 = time.time()
    pts, colors, ncc, _ = sweep(images, cams, K, D, box)
    n_px = (box[1] - box[0]) * (box[3] - box[2])
    print(f'{len(images)} images; {len(pts)} of {n_px} pixels in the '
          f'{box[1] - box[0]}x{box[3] - box[2]} box matched '
          f'(median NCC {np.median(ncc):.2f}) in {time.time() - t0:.1f}s')
    if len(pts) < 10:
        print('too few matches')
        return False
    lo, hi = pts.min(0), pts.max(0)
    print(f'extent: x {lo[0] * 1000:.1f}..{hi[0] * 1000:.1f}  '
          f'y {lo[1] * 1000:.1f}..{hi[1] * 1000:.1f}  '
          f'z {lo[2] * 1000:.1f}..{hi[2] * 1000:.1f} mm '
          f'(table at {cs.TABLE_Z * 1000:.1f})')
    write_ply(PLY_PATH, pts, colors)
    print(PLY_PATH)
    if show:
        import one.scene.scene_object_primitive as ossop
        import one.viewer.world as ovw
        c = np.median(pts, axis=0)
        base = ovw.World(cam_pos=c + np.array([0.25, 0.25, 0.25]),
                         cam_lookat_pos=c)
        ossop.point_cloud(pts, colors).attach_to(base.scene)
        base.run()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--no-show', action='store_true',
                        help='print and save only, no viewer')
    args = parser.parse_args()
    sys.exit(0 if main(show=not args.no_show) else 1)
