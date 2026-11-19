"""Track the chosen leaf's pitch live: the sampler goes to the close look
as ``suc_cut_store.py`` does (its steps 1-5), then, over and over, looks
at the leaf and turns the nozzle to its pitch -- tilt the pot by hand and
the nozzle follows. A test of the stereo's pitch (its J -> C axis), the
nozzle NOT over the leaf: from the second look's pose, the leaf clear of
the cutter in the picture (over it the cutter hides J).

    1-5 as suc_cut_store.py: search, find the plant, observe, look, to the
        close look (the eye CLOSE_LOOK_HEIGHT over the plant's top)
        (each after 'g')
    then, after 'g', until q / Esc in the terminal or the window closed:
        a frame; the leaves (SAM 3 once, 'leaf' -- the plant and both eyes
        were checked at the look); of those within TRACK_MAX_YAW of the
        last one's J -> C (the next leaf round is ~45 deg off), the one
        nearest its C (refused past TRACK_MAX_MOVE: lost, held); its pitch -- C over J -- the median
        of the last TRACK_MEDIAN, kept within TRACK_MAX_PITCH, the
        nozzle's turned toward it at most TRACK_PITCH_STEP; the second
        look's pose at that pitch (``leaf_pipeline.relook_goal``), the eye
        kept as high over the leaf as at the start -- none there (out of
        reach): at a smaller pitch (TRACK_FALLBACK of it); moved there at
        TRACK_SPEED when it is TRACK_MIN_TURN off -- a joint past
        TRACK_MAX_TURN: that far toward it (the next frame goes on), past
        TRACK_FLIP_TURN (the wrist flipped over): not; nor the tcp past
        TRACK_MAX_SHIFT

A window shows the frame's cloud and the leaf's frame (origin C, y along
J -> C at its pitch, z the nozzle's way in).

The collision model is suc_cut_store.py's -- not the plant: watch the
cutter's tip as the pot is tilted.

    py -3.12 coripps/tool/leaf/track_pitch.py
"""

import sys
import time
import threading

import numpy as np

from one.drivers.cobotta import BCapError

from coripps.tool.keys import _quit_pressed, _wait_for_g
from coripps.tool.leaf.suc_cut_store import CLOSE_LOOK_HEIGHT, LeafSession
from coripps.vision.sampler_stereo import leaf_pipeline as lp
from coripps.vision.sampler_stereo import leaves as lv


TRACK_SPEED = 20.0              # %
TRACK_MAX_MOVE = 0.020          # m, the leaf's C from one frame to the next
TRACK_MAX_YAW = np.deg2rad(20.0)    # ... and its J -> C about the vertical
                                    # (neighbours' C are only ~12-20 mm off)
TRACK_MEDIAN = 3                # frames the pitch is the median of
TRACK_MAX_PITCH = np.deg2rad(30.0)
TRACK_PITCH_STEP = np.deg2rad(10.0)  # the nozzle's pitch changed at most
                                     # this a frame: a jump followed in steps
TRACK_MIN_TURN = np.deg2rad(1.0)    # a joint less off than this: not moved
TRACK_MAX_SHIFT = 0.030         # m, the tcp moved further in one go: refused
TRACK_MAX_TURN = np.deg2rad(15.0)   # a joint turned at most this a move ...
TRACK_FLIP_TURN = np.deg2rad(45.0)  # ... further than this: refused (a flip)
TRACK_FALLBACK = (0.75, 0.5, 0.25, 0.0)    # the pitch's share tried when
                                           # it is out of reach
TRACK_SETTLE = 0.1              # s still before the next frame
# the window
CLOUD_STRIDE = 2                # every CLOUD_STRIDE-th pixel
FRAME_LENGTH = 0.03             # m, the drawn leaf frame
VIEW_OFFSET = (0.25, 0.15, 0.20)    # m, the view's eye from the leaf's C
UPDATE_INTERVAL = 0.1           # s


def leaf_pitch(leaf):
    """``leaf``'s J -> C axis over the horizontal (rad, + C higher)."""
    v = leaf['C'] - leaf['J']
    return float(np.arctan2(v[2], np.hypot(v[0], v[1])))


def yaw_off(a, b):
    """The angle (rad) between leaves ``a`` and ``b``'s J -> C about the
    vertical."""
    ya, yb = a['tf'][:2, 1], b['tf'][:2, 1]
    return abs(float(np.arctan2(ya[0] * yb[1] - ya[1] * yb[0], ya @ yb)))


def pitched_frame(leaf, pitch):
    """``leaf``'s frame turned up ``pitch`` (rad) about its x: y along J -> C
    at that pitch, z square to it (the nozzle's way in)."""
    up = np.array([0.0, 0.0, 1.0])
    yh = np.array([*leaf['tf'][:2, 1], 0.0])
    yh /= np.linalg.norm(yh)
    x = np.cross(up, yh)
    y = np.cos(pitch) * yh + np.sin(pitch) * up
    return lv.tf(leaf['C'], np.column_stack([x, y, np.cross(x, y)]))


def cloud(color, depth, K, cam_tf):
    """World points and RGB (0-1) of every CLOUD_STRIDE-th pixel with
    depth."""
    every = np.zeros(depth.shape, bool)
    every[::CLOUD_STRIDE, ::CLOUD_STRIDE] = True
    v, u = np.nonzero(every & (depth > 0))
    return (lv.depth_points(every, depth, K, cam_tf),
            color[v, u, ::-1] / 255.0)


def track(session, shown, stop):
    """Follow ``session.leaf``'s pitch until q / Esc or ``stop``; the latest
    cloud and leaf frame put in ``shown`` (under its 'lock')."""
    sampler, eye = session.sampler, session.eye
    leaf = session.leaf
    # the eye kept as high over the leaf as the close look puts it
    over_c = session.plant['plant_z'] - leaf['C'][2]
    pitches = []
    pitch = 0.0                     # the nozzle's: level at the start
    session.cobotta.enable(speed=TRACK_SPEED)
    while not stop.is_set() and not _quit_pressed():
        qs = session.qs_now()
        color, depth, K = eye.frame()
        cam_tf = lp.tf_w_cam(sampler, qs, session.tf_flange_cam)
        with shown['lock']:
            shown['cloud'] = cloud(color, depth, K, cam_tf)
        masks = list(session.segmenter.masks(lv.normalize(color), 'leaf'))
        found = lv.leaf_frames(color, depth, K, cam_tf, session.segmenter,
                               depth_of=eye.leaf_depth, leaf_masks=masks)
        if found is None or not found[0]:
            print('no leaves found')
            continue
        same = [f for f in found[0] if yaw_off(f, leaf) <= TRACK_MAX_YAW]
        if not same:
            print(f'no leaf within {np.degrees(TRACK_MAX_YAW):.0f} deg of the '
                  "tracked one's J -> C -- lost, held")
            continue
        near = min(same, key=lambda f: np.linalg.norm(f['C'] - leaf['C']))
        moved = np.linalg.norm(near['C'] - leaf['C'])
        if moved > TRACK_MAX_MOVE:
            print(f'the nearest leaf is {moved * 1000:.0f} mm off -- lost, '
                  'held')
            continue
        leaf = near
        pitches = (pitches + [leaf_pitch(leaf)])[-TRACK_MEDIAN:]
        target = float(np.clip(np.median(pitches), -TRACK_MAX_PITCH,
                               TRACK_MAX_PITCH))
        with shown['lock']:
            shown['frame'] = pitched_frame(leaf, target)
        step = pitch + float(np.clip(target - pitch, -TRACK_PITCH_STEP,
                                     TRACK_PITCH_STEP))
        y = leaf['tf'][:2, 1]
        print(f"pitch {np.degrees(pitches[-1]):+5.1f} (median "
              f"{np.degrees(target):+5.1f}) -> nozzle "
              f"{np.degrees(step):+5.1f} deg, yaw "
              f"{np.degrees(np.arctan2(y[1], y[0])):+6.1f} deg, C "
              f"{np.round(leaf['C'] * 1000, 1)} mm", end='')
        for share in (1.0,) + TRACK_FALLBACK:
            goal = lp.relook_goal(sampler, session.pln_ctx,
                                  session.tf_flange_cam, eye, leaf,
                                  leaf['C'][2] + over_c, qs, step * share,
                                  height=CLOSE_LOOK_HEIGHT)
            if goal is not None:
                break
        if goal is None:
            print(' -- out of reach, held')
            continue
        if share < 1.0:
            step *= share
            print(f' -- out of reach, at {np.degrees(step):+.1f} deg',
                  end='')
        turn = np.max(np.abs(goal - qs))
        if turn < TRACK_MIN_TURN:
            print()
            pitch = step
            continue
        if turn > TRACK_FLIP_TURN:
            print(f' -- a joint would turn {np.degrees(turn):.0f} deg (a '
                  'flip), held')
            continue
        if turn > TRACK_MAX_TURN:
            part = TRACK_MAX_TURN / turn
            goal = (qs + part * (goal - qs)).astype(np.float32)
            step = pitch + part * (step - pitch)
            print(f' -- {part:.0%} of the way', end='')
        sampler.fk(qs=qs)
        before = np.array(sampler.cutter.tcp('tcp').tf[:3, 3])
        sampler.fk(qs=goal)
        shift = np.linalg.norm(np.array(sampler.cutter.tcp('tcp').tf[:3, 3])
                               - before)
        if shift > TRACK_MAX_SHIFT:
            print(f' -- the tcp would move {shift * 1000:.0f} mm, held')
            continue
        if not session.pln_ctx.is_motion_valid(qs, goal):
            print(' -- in collision, held')
            continue
        print(f' -- moved {shift * 1000:.0f} mm')
        session.cobotta.move_path([goal])
        pitch = step
        time.sleep(TRACK_SETTLE)


def show(session):
    """:func:`track` in a thread, its cloud and leaf frame in a window, until
    either ends; True unless the track failed."""
    import pyglet
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    c = session.leaf['C']
    base = ovw.World(cam_pos=c + VIEW_OFFSET, cam_lookat_pos=c)
    shown = {'lock': threading.Lock(), 'cloud': None, 'frame': None}
    drawn = {'cloud': None, 'frame': None}
    stop = threading.Event()
    failed = []

    def run():
        try:
            track(session, shown, stop)
        except (RuntimeError, OSError, BCapError) as exc:
            failed.append(exc)

    def update(dt):
        if not thread.is_alive():
            pyglet.app.exit()
            return
        with shown['lock']:
            new = {k: shown[k] for k in drawn}
            shown['cloud'] = shown['frame'] = None
        if new['cloud'] is not None:
            new['cloud'] = ossop.point_cloud(new['cloud'][0].astype(
                np.float32), new['cloud'][1])
        if new['frame'] is not None:
            new['frame'] = ossop.frame_from_tf(
                new['frame'], length_scale=FRAME_LENGTH / 0.2,
                radius_scale=FRAME_LENGTH / 0.2)
        for k, obj in new.items():
            if obj is None:
                continue
            if drawn[k] is not None:
                drawn[k].detach_from(base.scene)
            obj.attach_to(base.scene)
            drawn[k] = obj

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    base.schedule_interval(update, interval=UPDATE_INTERVAL)
    try:
        base.run()
    finally:
        stop.set()
        thread.join()
    if failed:
        print(f'stopped: {failed[0]}')
    return not failed


def main():
    try:
        session = LeafSession()
    except Exception as exc:
        print(f'cannot start: {exc}')
        return False
    try:
        makers = [session.s_search, session.s_find_plant, session.s_observe,
                  session.s_look, session.s_close]
        for k, make in enumerate(makers, 1):
            what, run = make()
            if not _wait_for_g(f'step {k}/{len(makers)}: {what}'):
                print('stopped')
                return False
            run()
        if not _wait_for_g('track the leaf\'s pitch: tilt the pot by hand '
                           '(q / Esc in the terminal, or close the window, '
                           'stops)'):
            print('stopped')
            return False
        ok = show(session)
        print('stopped')
        return ok
    except (RuntimeError, OSError, BCapError) as exc:
        print(f'stopped: {exc}')
        return False
    finally:
        session.close()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
