"""A live view of a real run, to see what the cell is doing and why:

* the 3D viewer: the platform, both robots and the capper where they really
  are (read every FOLLOW_INTERVAL in a thread of its own), the tubes in the
  rack -- the one being worked on red, the ones done gone -- and the point
  cloud of the sampler's last look (world);
* a picture window (``WINDOW``): the last look's left picture, every leaf
  outlined (the chosen one green, the other candidates yellow, the ones not
  takeable grey) and numbered, and under it a table: each leaf's C, slope,
  ovalness, C - J, room and why it is not taken.

The run's threads only :func:`publish` (a look, the tubes); the viewer's
own tick (:meth:`Live.update`, main thread) draws what is new. With no
viewer running, publishing only keeps the last of each.

    live = Live(base, stage, capper, tube_points)   # single_routine._Stage
    live.start()                        # follows the arms, draws what comes
    monitor.publish_tubes(3, done=(0, 1, 2))        # from the run's thread
"""

import queue
import threading
import time

import cv2
import numpy as np

import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.utils.constant as ouc

WINDOW = 'look'
FOLLOW_INTERVAL = 0.1           # s, the real arms read
UPDATE_INTERVAL = 1 / 15.0      # s, new things drawn
CLOUD_STRIDE = 2                # px, every this many of the depth picture
PANEL_ROW = 18                  # px, the table's row height

_lock = threading.Lock()
_posted = {'look': None, 'look_n': 0, 'tubes': None, 'tubes_n': 0}


# -- what the run posts --------------------------------------------------------
def publish(image=None, cloud=None):
    """A look to show: ``image`` (BGR) in the picture window, ``cloud``
    ``(points (n, 3) world, BGR (n, 3) uint8)`` in the viewer (None: the
    last one kept)."""
    with _lock:
        _posted['look'] = (image, cloud)
        _posted['look_n'] += 1


def publish_tubes(current, done=()):
    """The tube being worked on (its index in the rack, None: none) and the
    ones done (gone from the rack)."""
    with _lock:
        _posted['tubes'] = (current, tuple(done))
        _posted['tubes_n'] += 1


def cloud_world(color, depth, K, cam_tf, stride=CLOUD_STRIDE):
    """The depth picture's points in the world, with their colours."""
    v, u = np.nonzero(depth[::stride, ::stride] > 0)
    v, u = v * stride, u * stride
    z = depth[v, u]
    pts = np.c_[(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]
    return pts @ cam_tf[:3, :3].T + cam_tf[:3, 3], color[v, u]


def look_image(color, leaves, chosen, rows, title, takeable=None):
    """``color`` with every leaf outlined (chosen green, takeable yellow,
    else grey), C (red), J (blue), C - J (yellow) and its number, the
    chosen one starred; ``rows`` (one text line per leaf, then any more)
    in a table under it; ``title`` on top."""
    img = color.copy()
    for k, leaf in enumerate(leaves):
        ok = True if takeable is None else takeable(leaf)
        rgb = ((0, 255, 0) if k == chosen else
               (0, 255, 255) if ok else (150, 150, 150))
        cnts, _ = cv2.findContours(leaf['mask'].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, rgb, 2 if k == chosen else 1)
        c = tuple(int(t) for t in np.round(leaf['C_px']))
        j = tuple(int(t) for t in np.round(leaf['J_px']))
        cv2.line(img, j, c, (0, 255, 255), 1)
        cv2.circle(img, c, 3, (0, 0, 255), -1)
        cv2.circle(img, j, 3, (255, 0, 0), -1)
        label = f'{k}*' if k == chosen else str(k)
        for colour, width in (((0, 0, 0), 3), (rgb, 1)):
            cv2.putText(img, label, (c[0] + 5, c[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, width)
    return with_table(img, rows, title)


def with_table(img, rows, title=''):
    """``img`` with ``title`` on top and ``rows`` (text) under it."""
    h, w = img.shape[:2]
    panel = np.full((PANEL_ROW * (len(rows) + 1) + 6, w, 3), 30, np.uint8)
    for i, row in enumerate(rows):
        cv2.putText(panel, row, (6, PANEL_ROW * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    out = np.vstack([img, panel])
    if title:
        for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(out, title[:110], (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, colour, width)
    return out


# -- the viewer's side -----------------------------------------------------------
class Live:
    """Keeps ``base`` (an ``ovw.World``) and ``stage``
    (``single_routine._Stage``) showing the real cell, and the picture
    window the last look. ``capper``: its exec (the dial its last move
    ended at), or None. ``tube_points``: the rack's tubes (each its cap's
    top, world), drawn standing there."""

    def __init__(self, base, stage, capper=None, tube_points=None,
                 keys=False):
        self.base, self.stage, self.capper = base, stage, capper
        # the keys typed in the picture window (and, ``keys``, 'g' / 'q' in
        # the viewer), for :meth:`key`; ``closed`` once the viewer is
        self.keys, self.viewer_keys, self.closed = queue.Queue(), keys, False
        self.cloud = None
        self.seen = {'look_n': 0, 'tubes_n': 0}
        self.qs = {}                    # robot -> its last read joints
        self.stop = threading.Event()
        self.tubes, self.gone = [], set()
        if tube_points is not None:
            import coripps.picker_motion as pkm
            import coripps.tool.tube.poses_io_pick as ppio
            model = osso.SceneObject.from_file(
                pkm._TUBE_PATH, collision_type=None,
                rgb=ouc.ExtendedColor.SILVER)
            for pt in tube_points:          # standing, its cap's top there
                tube = model.clone()
                tube.set_pos_rotmat(
                    pos=np.asarray(pt)[:3] + ppio.TUBE_POS_OFFSET,
                    rotmat=np.eye(3))
                tube.attach_to(base.scene)
                self.tubes.append(tube)

    def start(self, follow=True):
        """Draw what comes (and, ``follow``, read the real arms -- or later,
        by :meth:`follow`)."""
        if follow:
            self.follow()
        self.base.schedule_interval(lambda dt: self.update(),
                                    interval=UPDATE_INTERVAL)
        return self

    def follow(self):
        """From now on, the arms where they really are."""
        if not getattr(self, '_following', False):
            self._following = True
            threading.Thread(target=self._follow, daemon=True,
                             name='live-follow').start()

    def _follow(self):
        from coripps import single_routine as sr
        hosts = ((self.stage.picker, sr._PICKER_HOST),
                 (self.stage.sampler, sr._SAMPLER_HOST))
        while not self.stop.wait(FOLLOW_INTERVAL):
            for robot, host in hosts:
                try:
                    self.qs[robot] = sr.read_qs(host, robot.ndof)
                except Exception:       # not reachable now: kept where it was
                    pass

    def update(self):
        """The arms where they were last read, and what was posted since
        the last call (main thread)."""
        for robot, qs in list(self.qs.items()):
            robot.fk(qs=qs)
        deg = None if self.capper is None else self.capper.last_deg
        if deg is not None:
            self.stage.platform.capper.set_dial(deg)
        with _lock:
            look = (_posted['look'] if _posted['look_n'] != self.seen['look_n']
                    else None)
            tubes = (_posted['tubes']
                     if _posted['tubes_n'] != self.seen['tubes_n'] else None)
            self.seen = {'look_n': _posted['look_n'],
                         'tubes_n': _posted['tubes_n']}
        if look is not None:
            image, cloud = look
            if cloud is not None:
                if self.cloud is not None:
                    self.cloud.detach_from(self.base.scene)
                pts, bgr = cloud
                self.cloud = ossop.point_cloud(
                    np.asarray(pts, np.float32),
                    np.asarray(bgr, np.float32)[:, ::-1] / 255.0)
                self.cloud.attach_to(self.base.scene)
            if image is not None:
                cv2.imshow(WINDOW, image)
        if tubes is not None and self.tubes:
            current, done = tubes
            for k, tube in enumerate(self.tubes):
                if k in done:
                    if k not in self.gone:
                        tube.detach_from(self.base.scene)
                        self.gone.add(k)
                    continue
                tube.rgb = (ouc.BasicColor.RED if k == current
                            else ouc.ExtendedColor.SILVER)
        self.base.scene.dirty = True
        typed = cv2.waitKey(1)          # the picture window kept drawn
        if typed != -1 and typed & 0xFF != 0xFF:
            self.keys.put(chr(typed & 0xFF).lower())
        if self.viewer_keys:
            from pyglet.window import key
            im = self.base.input_manager
            for k, c in ((key.G, 'g'), (key.Q, 'q')):
                if im.is_key_pressed_edge(k):
                    self.keys.put(c)

    def key(self):
        """A key typed in the windows, or None (after a short wait) -- an
        ``idle`` for ``one.utils.keys._wait_for_g``; 'q' once the viewer is
        closed."""
        if self.closed:
            return 'q'
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            time.sleep(0.03)
            return None
