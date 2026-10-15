"""The RealSense D405 on the cutter hand, by serial number.

Everything is given in the DEPTH frame: the D405's depth is rectified (pure
pinhole, no distortion) and its left infrared image IS that frame, so one K
serves all three images -- no lens calibration of our own:
  - infrared: the left imager, grey (the board for the hand-eye)
  - depth: metres, raw (no RealSense post-processing filters)
  - color: the colour image re-sampled into the depth frame
    (``rs.align(rs.stream.depth)``); black where there is no depth

    py -3.12 coripps/vision/d405/camera.py      # one frame, a few numbers
"""

import os
import json

import numpy as np
import pyrealsense2 as rs


SERIAL = '427622271640'         # the cutter hand's D405
WIDTH, HEIGHT, FPS = 640, 480, 30
FLUSH_FRAMES = 10               # frames dropped so the one returned is fresh
INTRINSICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'config', 'd405_intrinsics.json')


class D405:
    """The camera, open until :meth:`close`."""

    def __init__(self, serial=SERIAL):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,
                             FPS)
        config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT,
                             rs.format.y8, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8,
                             FPS)
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(f'D405 {serial}: cannot start ({exc})')
        self.depth_scale = (profile.get_device().first_depth_sensor()
                            .get_depth_scale())
        intr = (profile.get_stream(rs.stream.depth).as_video_stream_profile()
                .get_intrinsics())
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]])
        self.D = np.zeros(5)            # rectified
        self.align = rs.align(rs.stream.depth)

    def _frameset(self):
        """A fresh frameset, NOT aligned: depth and infrared are already in
        the depth frame (aligning would re-map the infrared through the
        depth too -- holes where depth is missing, shifts where it is
        noisy)."""
        for _ in range(FLUSH_FRAMES):
            frames = self.pipeline.wait_for_frames()
        return frames

    def frames(self):
        """``(color BGR, depth m float32, infrared grey)``, taken now, all
        in the depth frame (only color is re-sampled into it)."""
        frames = self._frameset()
        depth = (np.asanyarray(frames.get_depth_frame().get_data())
                 .astype(np.float32) * self.depth_scale)
        infrared = np.asanyarray(frames.get_infrared_frame(1).get_data()).copy()
        color = np.asanyarray(self.align.process(frames).get_color_frame()
                              .get_data()).copy()
        return color, depth, infrared

    def frame(self):
        """The infrared image as BGR (the camera interface of the hand-eye
        session: the board is found in it)."""
        infrared = np.asanyarray(self._frameset().get_infrared_frame(1)
                                 .get_data())
        return np.repeat(infrared[:, :, None], 3, axis=2)

    def save_intrinsics(self, path=INTRINSICS_PATH):
        """K (the depth frame's) to ``path``, for solving without the
        camera."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'serial': SERIAL, 'image_size': [WIDTH, HEIGHT],
                       'K': self.K.tolist(), 'D': self.D.tolist()}, f,
                      indent=2)
        return path

    def close(self):
        self.pipeline.stop()


def load_intrinsics(path=INTRINSICS_PATH):
    """``(K, D)`` saved by :meth:`D405.save_intrinsics`."""
    with open(path, encoding='utf-8') as f:
        cal = json.load(f)
    return np.array(cal['K']), np.array(cal['D'])


if __name__ == '__main__':
    cam = D405()
    try:
        color, depth, infrared = cam.frames()
        valid = depth[depth > 0]
        print(f'D405 {SERIAL}: K fx {cam.K[0, 0]:.2f} fy {cam.K[1, 1]:.2f} '
              f'cx {cam.K[0, 2]:.2f} cy {cam.K[1, 2]:.2f}')
        print(f'color {color.shape}, infrared {infrared.shape}, depth valid '
              f'{100 * valid.size / depth.size:.0f}%, median '
              f'{np.median(valid):.3f} m')
        print(cam.save_intrinsics())
    finally:
        cam.close()
