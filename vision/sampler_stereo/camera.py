"""The sampler's two webcams, left and right, by hub port.

Both are the "VGA USB Camera" model (640x480) with 1.7 mm fisheye lenses
and no IR-cut filter (they see the 940 nm light; colours are off under
room light). They sit on ports 1 and 3 of one hub, the hub on port 4 of
the main hub -- the picker's camera is on a port 3 too, of another hub (on
the main hub's port 3), so the ports are given with their hub's
(``cameras.index_of`` matches the port path from the camera up). Only the
hubs' own wiring is in them, not where the main hub plugs into the PC:
another PC, another USB port, the same ports.

The two are separate USB cameras, not synchronised: a thread of
:class:`StereoCam` grabs both, then decodes both, over and over, and keeps
the newest pair -- the two a few ms apart, and no stale frames to flush
when one is asked for (that took ~0.7 s a pair).

Their auto exposure stretches to ~90 ms (10.7 fps) in the dark and is
not steady (the same view 53 or 105 a moment apart). Nor is a manual
exposure changed while running: a step's brightness then depends on the
steps before it (up to ~3x: -7 is 30 coming down, 83 coming back up --
some slow gain of their own), so no exposure bracket (HDR) can be taken
with them. Set once and left alone it holds (to 0.1 of 255, opening after
opening): :class:`StereoCam` sets EXPOSURE (DirectShow's steps, 2^e s)
when it opens and never again -- a different light (940 nm) wants a
different EXPOSURE, which :meth:`StereoCam.frames` warns of. The two
cameras are not alike at one step (the right ~1.4x as bright): the depth
evens them (``depth.disparity``).
"""

import threading
import time

import cv2
import numpy as np

from one.vision.calibrate.handeye_capture import WebCam


PORTS = {'left': (1, 4), 'right': (3, 4)}   # port, its hub's port
EXPOSURE = -7                   # set once when opened (2^-7 s), never again
SETTLE_READS = 10               # frames read after setting it, at most ...
SETTLE_REL = 0.03               # ... until two in a row agree this well
WHITE = 235                     # grey counted clipped (the cameras top at ~237)
# a picture's middle half clipped past CLIPPED_WARN, or a mean under
# DARK_WARN: the light is not EXPOSURE's -- said so
CLIPPED_WARN = 0.05
DARK_WARN = 10.0
FRAMES_TIMEOUT = 2.0            # s, waited at most for a pair taken now
LABELS = {side: f'sampler {side} camera' for side in PORTS}
WINDOW = 'sampler stereo (q quits)'


class StereoCam:
    """Both cameras, open until :meth:`close`, at ``exposure``; a thread
    keeps the newest pair."""

    def __init__(self, exposure=EXPOSURE):
        self.cams = {}
        self._pair = None               # (time its grab began, (left, right))
        self._error = None
        self._new = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        try:
            for side in PORTS:
                self.cams[side] = WebCam(PORTS[side], LABELS[side])
                self._expose(side, exposure)
        except RuntimeError:
            self.close()
            raise
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self):
        caps = [self.cams[side].cap for side in PORTS]
        while not self._stop.is_set():
            t = time.perf_counter()
            pics = []
            ok = all(cap.grab() for cap in caps)
            for cap in caps if ok else ():
                ok, pic = cap.retrieve()
                if not ok:
                    break
                pics.append(pic)
            with self._new:
                if ok:
                    self._pair = t, tuple(pics)
                else:
                    self._error = 'a camera gives no image'
                self._new.notify_all()
            if not ok:
                return

    def _expose(self, side, step):
        """Manual exposure ``step`` on ``side``, frames read until two in a
        row agree in brightness."""
        cap = self.cams[side].cap
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)            # manual
        cap.set(cv2.CAP_PROP_EXPOSURE, step)
        last = None
        for _ in range(SETTLE_READS):
            ok, pic = cap.read()
            if not ok:
                raise RuntimeError(f'{LABELS[side]}: gives no image')
            mean = float(pic.mean())
            if last is not None and abs(mean - last) <= SETTLE_REL * max(
                    last, 1.0):
                return
            last = mean

    def frame(self, side):
        """``side``'s picture (BGR), taken now."""
        return self.frames()[list(PORTS).index(side)]

    def frames(self, warn=True, now=True):
        """``(left, right)`` (BGR), taken now -- the first pair the thread
        began to grab after this call (e.g. after the arm moved); without
        ``now``, the newest pair, up to a frame old, not waited for (a
        live view). With ``warn``, a warning when the light is not the
        exposure's (clipped or dark)."""
        asked = time.perf_counter() if now else -np.inf
        with self._new:
            if not self._new.wait_for(
                    lambda: self._error is not None or (
                        self._pair is not None and self._pair[0] >= asked),
                    timeout=FRAMES_TIMEOUT):
                raise RuntimeError(f'no picture in {FRAMES_TIMEOUT} s')
            if self._error is not None:
                raise RuntimeError(self._error)
            pair = self._pair[1]
        for side, pic in zip(PORTS, pair if warn else ()):
            grey = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            h, w = grey.shape
            clipped = (grey[h // 4:3 * h // 4, w // 4:3 * w // 4]
                       > WHITE).mean()
            if clipped > CLIPPED_WARN or grey.mean() < DARK_WARN:
                print(f'{LABELS[side]}: {clipped:.0%} of the middle clipped, '
                      f'mean {grey.mean():.0f} -- the light wants another '
                      'EXPOSURE')
        return pair

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        for cam in self.cams.values():
            cam.close()
        self.cams = {}


def live():
    """Both pictures side by side, left | right, until q / Esc. True
    unless a camera failed."""
    try:
        cam = StereoCam()
    except RuntimeError as exc:
        print(exc)
        return False
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            try:
                pics = [pic.copy()
                        for pic in cam.frames(warn=False, now=False)]
            except RuntimeError as exc:
                print(exc)
                return False
            for side, pic in zip(PORTS, pics):
                cv2.putText(pic, side, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            cv2.imshow(WINDOW, np.hstack(pics))
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                return True
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return True
    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    live()