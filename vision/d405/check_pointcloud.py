"""Live check of the D405 on the cutter hand: the robots at their real joint
angles, the D405's point cloud put in the world with the sampler's FK and
the hand-eye tf (``coripps/config/d405_handeye.json``), and the D405's
frame (its depth frame) drawn.

Jog the sampler: with a good hand-eye the cloud of the table and what is on
it stays put in the world while the camera moves. Read-only -- nothing is
sent to the robots.

    py -3.12 coripps/vision/d405/check_pointcloud.py
"""

import json

import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.drivers.cobotta import BCapError, Cobotta

from coripps.robots.mobisys import mobisys
from coripps.tool.home_sampler import _SAMPLER_HOST, _PICKER_HOST
from coripps.vision.d405.camera import D405
from coripps.vision.d405.calib import HANDEYE_PATH


UPDATE_INTERVAL = 0.2           # s
STRIDE = 2                      # every STRIDE-th pixel in u and v
DEPTH_MIN, DEPTH_MAX = 0.07, 1.0    # m, kept depth range
FRAME_LENGTH = 0.05             # m, drawn D405 frame


def connect(host, label):
    try:
        cobotta = Cobotta(host).connect()
    except (OSError, BCapError) as exc:
        print(f'{label}: cannot read {host} ({exc}) -- left at its sim pose')
        return None
    print(f'{label}: reading {host}')
    return cobotta


def main():
    with open(HANDEYE_PATH, encoding='utf-8') as f:
        tf_flange_cam = np.array(json.load(f)['tf_flange_cam'])
    cam = D405()
    K = cam.K
    h, w = 480, 640
    vv, uu = np.mgrid[0:h:STRIDE, 0:w:STRIDE]
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                     np.ones(uu.shape)], -1).reshape(-1, 3)

    base = ovw.World(cam_pos=(1.0, 0.2, 0.7),
                     cam_lookat_pos=(0.35, -0.15, 0.10))
    mobisys_base, picker, sampler = mobisys()
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    links = [(picker, connect(_PICKER_HOST, 'picker')),
             (sampler, connect(_SAMPLER_HOST, 'sampler'))]
    shown = {'cloud': None, 'frame': None}

    def update(dt):
        for robot, cobotta in links:
            if cobotta is not None:
                try:
                    robot.fk(qs=cobotta.joint_angles(ndof=robot.ndof))
                except (OSError, BCapError):
                    pass
        frames = cam.pipeline.poll_for_frames()
        if not frames:
            return
        depth_frame = frames.get_depth_frame()
        color_frame = cam.align.process(frames).get_color_frame()
        if not depth_frame or not color_frame:
            return
        depth = (np.asanyarray(depth_frame.get_data())[::STRIDE, ::STRIDE]
                 .astype(np.float64) * cam.depth_scale).ravel()
        color = np.asanyarray(color_frame.get_data())[::STRIDE, ::STRIDE]
        keep = (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
        tf_w_cam = (np.array(sampler.tcp('flange').tf, dtype=np.float64)
                    @ tf_flange_cam)
        pts = (rays[keep] * depth[keep, None]) @ tf_w_cam[:3, :3].T \
            + tf_w_cam[:3, 3]
        rgb = color.reshape(-1, 3)[keep][:, ::-1] / 255.0
        for name in ('cloud', 'frame'):
            if shown[name] is not None:
                shown[name].detach_from(base.scene)
        shown['cloud'] = ossop.point_cloud(pts.astype(np.float32), rgb)
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
                cobotta.close()


if __name__ == '__main__':
    main()
