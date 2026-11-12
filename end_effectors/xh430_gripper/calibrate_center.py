"""Calibrate the gripper's grasp center (``_GRASP_CENTER`` in xh430_gripper)
on the picker, by hand-aligning the fingers' centre to one fixed point from
several hand orientations.

    py -3.12 coripps/end_effectors/xh430_gripper/calibrate_center.py

In the viewer's panel:

1. "go to the start pose": the flange to START_POS, its x along world -y
   and its z along world -z (up to a safe height, turned, across, down).
2. Jog the arm (world x / y / z) until the fingers' centre sits on the
   point -- any fixed point; where it is does not matter, it is solved for.
3. "record + turn 45": the pose is recorded and the hand turned 45 deg
   about the flange's vertical axis to the next orientation (first the way
   J6 has more room, until its limit, then the other side) -- go to 2.
   Every orientation gets one sample; ~7 fit in J6's range.

After every sample the fit is redone. The fingers' centre is a fixed point
``c`` in the flange frame (the gripper's root frame IS the flange's) and
lands on the same world point ``P`` each time, so every sample ``(p, R)``
gives ``p + R c = P`` -- least squares over the samples for ``c_xy`` and
``P_xy`` (4 unknowns, 2 equations a sample: 3 samples or more). Only the
recorded poses enter, never the start pose. ``c_z`` is not observable (all
the turns are about the vertical), so it stays as it is. The panel shows
``c``, ``P``, each sample's residual (how well it was aligned), and the
suggested ``_GRASP_CENTER``, to copy -- the source is not written.

The samples are saved to SAMPLES_PATH after each record. No collision
check (as jog_tcp): the one at the controls watches the arm.
"""

import os
import sys
import json
import time
import threading

import numpy as np

import one.utils.math as oum
from one.drivers.cobotta import BCapError, CobottaMotion

from coripps.tool import pose_sequence as pseq
from coripps.tool.jog_tcp import (tcp_tf, jog_path, turn_path, read_qs,
                                  _PICKER_HOST, _SAMPLER_HOST)
import coripps.end_effectors.xh430_gripper.xh430_gripper as xg


START_POS = np.array([0.150, -0.060, 0.142])    # m, the flange
# flange x = world -y, z = world -z (so y = world -x)
START_ROTMAT = np.column_stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).astype(
    np.float64)
SAFE_Z = 0.200                  # m, the flange's height for the way there
TURN_DEG = 45.0                 # between two orientations
MAX_SAMPLES = 8                 # 360 / TURN_DEG
SPEED = 20                      # %, to begin with
MAX_SPEED = 30
STEPS_MM = (0.05, 0.1, 0.5, 1, 5)
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'calibrate_center_samples.json')


def pose_path(mech, tcp, qs, pos, rotmat, step=0.001, turn=np.radians(1)):
    """IK path from ``qs`` taking ``tcp`` straight to ``(pos, rotmat)``
    (position linear, orientation slerp), ``qs`` first; None when there is
    no smooth IK."""
    tf = tcp_tf(mech, tcp, qs)
    p0, r0 = tf[:3, 3], tf[:3, :3]
    _, angle = oum.axangle_between_rotmat(r0, rotmat)
    n = max(1, int(np.ceil(np.linalg.norm(pos - p0) / step)),
            int(np.ceil(angle / turn)))
    rots = oum.rotmat_slerp(r0, rotmat, n + 1)
    path = [np.asarray(qs, dtype=np.float32)]
    for k in range(1, n + 1):
        nxt = pseq.follow_positions(
            mech, path[-1], [(p0 + (pos - p0) * k / n).astype(np.float32)],
            rots[k].astype(np.float32), tcp=tcp)
        if nxt is None:
            return None
        path.append(nxt[-1])
    return np.asarray(path, dtype=np.float32)


def yaw_deg(rotmat):
    """Where the frame's x points, about world z (deg)."""
    return float(np.degrees(np.arctan2(rotmat[1, 0], rotmat[0, 0])))


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def fit(samples, c_z):
    """``(c_xy, P_xy, residuals)`` from the samples (each ``{'p', 'R'}``,
    the flange in the world): ``p + R c = P`` in xy, ``c_z`` given."""
    rows, rhs = [], []
    for s in samples:
        p, r = np.asarray(s['p']), np.asarray(s['R'])
        rows.append(np.hstack([r[:2, :2], -np.eye(2)]))
        rhs.append(-p[:2] - r[:2, 2] * c_z)
    x, *_ = np.linalg.lstsq(np.vstack(rows), np.hstack(rhs), rcond=None)
    c, P = x[:2], x[2:]
    res = [float(np.linalg.norm(np.asarray(s['p'])[:2]
                                + np.asarray(s['R'])[:2, :2] @ c
                                + np.asarray(s['R'])[:2, 2] * c_z - P))
           for s in samples]
    return c, P, res


def main():
    from imgui_bundle import imgui
    import one.viewer.world as ovw
    import one.viewer.imgui_overlay as ovio
    from coripps.robots.mobisys import mobisys

    _, mech, _ = mobisys()                      # for the IK (move thread)
    mobisys_base, picker, sampler = mobisys()   # on screen
    flange = mech.tcp('flange')
    c_z = float(xg._GRASP_CENTER[2])
    try:
        motion = CobottaMotion(_PICKER_HOST, speed=SPEED).connect()
    except (OSError, BCapError) as exc:
        print(f'picker: cannot reach {_PICKER_HOST} ({exc})')
        return False
    sampler_qs = read_qs(_SAMPLER_HOST, sampler.ndof)
    if sampler_qs is not None:
        sampler.fk(qs=sampler_qs)
    state = {'qs': np.float32(motion.joint_angles(ndof=mech.ndof)),
             'busy': False, 'speed': SPEED, 'step': 0.0005, 'msg': 'ready',
             'bad': False, 'samples': [], 'turned': 0.0, 'dir': None,
             'done': False}
    picker.fk(qs=state['qs'])

    def say(msg, bad=False):
        state['msg'], state['bad'] = msg, bad
        print(f'  {msg}')

    def play(path):
        motion.enable(speed=state['speed'])
        motion.set_speed(state['speed'])
        motion.move_path(path[1:])
        state['qs'] = np.float32(motion.joint_angles(ndof=mech.ndof))

    def run(job, *args):
        try:
            job(*args)
        except (OSError, BCapError) as exc:
            say(f'picker: {exc}', bad=True)
        finally:
            state['busy'] = False

    def start(job, *args, msg='moving ...'):
        if state['busy']:
            return
        state['busy'], state['msg'], state['bad'] = True, msg, False
        threading.Thread(target=run, args=(job, *args), daemon=True).start()

    def go_start():
        tf = tcp_tf(mech, flange, state['qs'])
        here, rot = tf[:3, 3], tf[:3, :3]
        up = max(SAFE_Z, here[2])
        legs = (('up', np.array([here[0], here[1], up]), rot),
                ('turn', np.array([here[0], here[1], up]), START_ROTMAT),
                ('across', np.array([*START_POS[:2], up]), START_ROTMAT),
                ('down', START_POS, START_ROTMAT))
        for name, pos, r in legs:
            path = pose_path(mech, flange, state['qs'], pos, r)
            if path is None:
                say(f'to the start pose: no smooth IK ({name}) -- stopped',
                    bad=True)
                return
            play(path)
        say('at the start pose: align the fingers\' centre on your point, '
            'then "record + turn 45"')

    def jog(axis, sign):
        path, why = jog_path(mech, flange, state['qs'], axis,
                             sign * state['step'], world=True)
        if path is None:
            say(why, bad=True)
            return
        play(path)
        say(f"{'xyz'[axis]}{'+' if sign > 0 else '-'} "
            f"{state['step'] * 1000:g} mm")

    def save():
        with open(SAMPLES_PATH, 'w', encoding='utf-8') as f:
            json.dump({'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'grasp_center_z': c_z,
                       'samples': state['samples']}, f, indent=1)

    def record_and_turn():
        qs = np.float32(motion.joint_angles(ndof=mech.ndof))
        state['qs'] = qs
        tf = tcp_tf(mech, flange, qs)
        if state['samples'] and any(
                abs(wrap180(state['turned'] - s['turned'])) < 1.0
                for s in state['samples']):
            say('this orientation is already recorded', bad=True)
            return
        state['samples'].append({
            'qs_deg': np.degrees(qs).round(4).tolist(),
            'p': tf[:3, 3].tolist(), 'R': tf[:3, :3].tolist(),
            'yaw_deg': yaw_deg(tf[:3, :3]), 'turned': state['turned']})
        save()
        n = len(state['samples'])
        if n >= MAX_SAMPLES:
            state['done'] = True
            say(f'sample {n} recorded -- all orientations done')
            return
        # the next orientation not yet sampled: on the side J6 has more room
        # first, then the other side
        if state['dir'] is None:
            j6 = np.degrees(qs[5])
            # a turn of +a about world z (tool z down) takes J6 by -a
            state['dir'] = 1.0 if j6 + 170 > 170 - j6 else -1.0
        sampled = [s['turned'] for s in state['samples']]
        for side in (state['dir'], -state['dir']):
            for k in range(1, MAX_SAMPLES):
                target = side * TURN_DEG * k
                if any(abs(wrap180(target - t)) < 1.0 for t in sampled):
                    continue
                path, _ = turn_path(mech, flange, qs,
                                    np.radians(target - state['turned']),
                                    world=True)
                if path is None:
                    break               # past J6's limit on this side
                play(path)
                state['turned'] = target
                say(f'sample {n} recorded; turned to {target:+.0f} deg -- '
                    'align again, then "record + turn 45"')
                return
        state['done'] = True
        say(f'sample {n} recorded -- no orientation left that J6 reaches')

    def panel():
        w = imgui.get_frame_height() * 2.6
        imgui.set_next_window_pos((20, 20), imgui.Cond_.first_use_ever)
        imgui.begin('calibrate grasp center', None,
                    imgui.WindowFlags_.always_auto_resize)
        busy = state['busy']
        imgui.begin_disabled(busy)
        if imgui.button('1. go to the start pose'):
            start(go_start)
        imgui.end_disabled()
        imgui.separator()
        imgui.text('2. align: jog (world)')
        imgui.begin_disabled(busy)
        for axis, label in enumerate('XYZ'):
            if imgui.button(f'{label}-', (w, 0)):
                start(jog, axis, -1.0)
            imgui.same_line()
            if imgui.button(f'{label}+', (w, 0)):
                start(jog, axis, 1.0)
        imgui.end_disabled()
        imgui.text('step mm')
        for mm in STEPS_MM:
            imgui.same_line()
            if imgui.radio_button(f'{mm:g}', bool(np.isclose(
                    state['step'] * 1000, mm))):
                state['step'] = mm / 1000
        _, state['speed'] = imgui.slider_int('speed %', state['speed'], 1,
                                             MAX_SPEED)
        imgui.separator()
        imgui.begin_disabled(busy or state['done'])
        if imgui.button('3. record + turn 45'):
            start(record_and_turn, msg='recording ...')
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(busy or not state['samples'])
        if imgui.button('undo last sample'):
            s = state['samples'].pop()
            state['done'] = False
            save()
            say(f"sample {len(state['samples']) + 1} removed (the arm stays "
                f"where it is, turned {state['turned']:+.0f} deg)")
        imgui.end_disabled()
        # where the flange is
        tf = tcp_tf(mech, flange, state['qs'])
        p = tf[:3, 3] * 1000
        imgui.text(f'flange {p[0]:8.2f} {p[1]:8.2f} {p[2]:8.2f} mm, '
                   f'yaw {yaw_deg(tf[:3, :3]):+.1f} deg, turned '
                   f"{state['turned']:+.0f}")
        # the samples and the fit
        samples = state['samples']
        if samples:
            imgui.separator()
            fitted = len(samples) >= 3
            if fitted:
                c, P, res = fit(samples, c_z)
            for i, s in enumerate(samples):
                sp = np.asarray(s['p']) * 1000
                line = (f"{i + 1}: turned {s['turned']:+5.0f}  flange "
                        f'{sp[0]:7.2f} {sp[1]:7.2f}')
                if fitted:
                    line += f'  residual {res[i] * 1000:.2f} mm'
                imgui.text(line)
            if fitted:
                imgui.separator()
                rms = np.sqrt(np.mean(np.square(res))) * 1000
                imgui.text(f'centre in the flange frame: x {c[0] * 1000:+.2f} '
                           f'y {c[1] * 1000:+.2f} mm (z {c_z * 1000:g}, kept)')
                imgui.text(f'aligned point P: {P[0] * 1000:.2f} '
                           f'{P[1] * 1000:.2f} mm;  residual rms {rms:.2f}, '
                           f'max {max(res) * 1000:.2f} mm')
                old = np.asarray(xg._GRASP_CENTER) * 1000
                imgui.text(f'_GRASP_CENTER now ({old[0]:+.2f}, '
                           f'{old[1]:+.2f}, {old[2]:g}) mm')
                line = (f'_GRASP_CENTER = ({c[0]:.5f}, {c[1]:.5f}, '
                        f'{c_z:g})')
                imgui.text(line)
                if imgui.button('copy'):
                    imgui.set_clipboard_text(line)
                    print(line)
                    say('copied to the clipboard')
            else:
                imgui.text_disabled(f'{3 - len(samples)} more sample(s) '
                                    'for a fit')
        imgui.separator()
        if state['bad']:
            imgui.text_colored((1.0, 0.35, 0.3, 1.0), state['msg'])
        else:
            imgui.text(state['msg'])
        imgui.end()

    centre = tcp_tf(mech, flange, state['qs'])[:3, 3]
    base = ovw.World(cam_pos=centre + np.array([0.6, 0.45, 0.35]),
                     cam_lookat_pos=centre)
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    ovio.ImGuiOverlay(base, panel)

    def tick(dt):
        picker.fk(qs=state['qs'])

    base.schedule_interval(tick, interval=1 / 30)
    try:
        base.run()
    finally:
        motion.disconnect()             # motors off, arm given back
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
