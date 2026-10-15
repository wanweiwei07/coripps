"""Jog a robot's tcp along the tcp's OWN axes, in the viewer, by chords.

In the viewer window (it has to have the focus):

    hold x / y / z, tap + or -   one STEP along the tcp's x / y / z
                                 (+: the '=/+' key or keypad +;
                                  -: the '-' key or keypad -)
    [ / ]                        halve / double STEP (MIN_STEP..MAX_STEP)
    p                            print the tcp pose
    close the window             quit

One step per tap of + / -; keys are ignored while the arm moves. The
viewer shows the cell with the jogged robot at its real joints and its tcp
frame; the other robot is read once.

The sampler jogs its cutter's 'tcp', the picker its gripper's
'grasp_center'. Each move is a straight line (IK nearest the last, every
SAMPLE apart), collision-checked against the other robot (read once,
frozen) and the ground before it runs; nothing moves when it fails. The arm
is enabled at the first move and given back when the window closes.

    py -3.12 coripps/tool/jog_tcp.py                 # the sampler, 1 mm
    py -3.12 coripps/tool/jog_tcp.py --robot picker --step 5
"""

import sys
import argparse
import threading

import numpy as np

from one.drivers.cobotta import BCapError, Cobotta, CobottaMotion

from coripps.tool import pose_sequence as pseq


SPEED = 10.0                    # %
SAMPLE = 0.001                  # m between a move's IK samples
MIN_STEP, MAX_STEP = 0.0001, 0.020      # m
TICK = 1 / 30.0                 # s
FRAME_LENGTH = 0.05             # m, the drawn tcp frame
_SAMPLER_HOST = '192.168.0.101'
_PICKER_HOST = '192.168.0.102'


def build(robot):
    """``(mech, tcp, pln_ctx, host)`` of the robot to jog."""
    if robot == 'sampler':
        from coripps.tool.home_sampler import _build
        _, _, mech, pln_ctx = _build()
        return mech, mech.cutter.tcp('tcp'), pln_ctx, _SAMPLER_HOST
    from coripps.tool.tube.home_picker import _build
    _, _, mech, pln_ctx = _build()
    return mech, mech.gripper.tcp('grasp_center'), pln_ctx, _PICKER_HOST


def tcp_tf(mech, tcp, qs):
    mech.fk(qs=qs)
    return np.array(tcp.tf, dtype=np.float64)


def print_pose(mech, tcp, qs):
    tf = tcp_tf(mech, tcp, qs)
    print(f'  tcp {np.round(tf[:3, 3] * 1000, 2)} mm  x '
          f'{np.round(tf[:3, 0], 3)}  y {np.round(tf[:3, 1], 3)}  z '
          f'{np.round(tf[:3, 2], 3)}  joints {np.round(np.degrees(qs), 2)} '
          'deg')


def jog_path(mech, tcp, pln_ctx, qs, axis, dist):
    """A straight path moving the tcp ``dist`` along its own ``axis``
    (0 / 1 / 2), ``qs`` first; None (said why) when it cannot."""
    tf = tcp_tf(mech, tcp, qs)
    delta = dist * tf[:3, axis]
    n = max(int(np.ceil(abs(dist) / SAMPLE)), 1)
    path = pseq.follow_positions(
        mech, qs, [(tf[:3, 3] + delta * k / n).astype(np.float32)
                   for k in range(1, n + 1)],
        tf[:3, :3].astype(np.float32), tcp=tcp)
    if path is None:
        print('  no smooth IK for that move -- not moved')
        return None
    if not all(pln_ctx.is_motion_valid(a, b)
               for a, b in zip(path[:-1], path[1:])):
        print('  that move collides -- not moved')
        return None
    return path


def read_qs(host, ndof):
    try:
        with Cobotta(host) as reader:
            return reader.joint_angles(ndof=ndof)
    except (OSError, BCapError) as exc:
        print(f'{host}: cannot read ({exc}) -- shown at its sim pose')
        return None


def main(robot='sampler', step=0.001):
    from pyglet.window import key
    import one.viewer.world as ovw
    from coripps.robots.mobisys import mobisys

    mech, tcp, pln_ctx, host = build(robot)
    try:
        cobotta = CobottaMotion(host, speed=SPEED).connect()
    except (OSError, BCapError) as exc:
        print(f'{robot}: cannot reach {host} ({exc})')
        return False

    # the cell on screen: the jogged robot follows its real joints
    mobisys_base, picker, sampler = mobisys()
    shown, other, other_host = ((sampler, picker, _PICKER_HOST)
                                if robot == 'sampler'
                                else (picker, sampler, _SAMPLER_HOST))
    other_qs = read_qs(other_host, other.ndof)
    if other_qs is not None:
        other.fk(qs=other_qs)
    state = {'qs': pln_ctx.enforce_bounds(cobotta.joint_angles(
        ndof=mech.ndof)), 'step': step, 'busy': False}
    shown.fk(qs=state['qs'])
    centre = tcp_tf(mech, tcp, state['qs'])[:3, 3]
    base = ovw.World(cam_pos=centre + np.array([0.6, 0.45, 0.35]),
                     cam_lookat_pos=centre)
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    scale = FRAME_LENGTH / 0.2          # the frame primitive is 0.2 m long
    tool = sampler.cutter if robot == 'sampler' else picker.gripper
    tool.toggle_tcp('tcp' if robot == 'sampler' else 'grasp_center',
                    length_scale=scale, radius_scale=scale)
    print(__doc__.split('\n\n')[1])
    print_pose(mech, tcp, state['qs'])
    print(f'step {state["step"] * 1000:g} mm')

    axis_keys = {'x': key.X, 'y': key.Y, 'z': key.Z}
    plus_keys = (key.EQUAL, key.PLUS, key.NUM_ADD)
    minus_keys = (key.MINUS, key.NUM_SUBTRACT)

    def move(axis, sign):
        try:
            path = jog_path(mech, tcp, pln_ctx, state['qs'], axis,
                            sign * state['step'])
            if path is not None:
                cobotta.enable(speed=SPEED)
                cobotta.move_path(path[1:])
                print(f"{'xyz'[axis]}{'+' if sign > 0 else '-'} "
                      f"{state['step'] * 1000:g} mm")
            state['qs'] = pln_ctx.enforce_bounds(
                cobotta.joint_angles(ndof=mech.ndof))
            if path is not None:
                print_pose(mech, tcp, state['qs'])
        except (OSError, BCapError) as exc:
            print(f'{robot}: {exc}')
        finally:
            state['busy'] = False

    def tick(dt):
        im = base.input_manager
        # read every edge each tick, so none is left over for later
        plus = any([im.is_key_pressed_edge(k) for k in plus_keys])
        minus = any([im.is_key_pressed_edge(k) for k in minus_keys])
        smaller = im.is_key_pressed_edge(key.BRACKETLEFT)
        larger = im.is_key_pressed_edge(key.BRACKETRIGHT)
        show = im.is_key_pressed_edge(key.P)
        shown.fk(qs=state['qs'])
        if state['busy']:
            return
        if smaller or larger:
            state['step'] = float(np.clip(
                state['step'] * (0.5 if smaller else 2.0), MIN_STEP,
                MAX_STEP))
            print(f'step {state["step"] * 1000:g} mm')
        if show:
            print_pose(mech, tcp, state['qs'])
        held = [a for a, k in axis_keys.items() if im.is_key_pressed(k)]
        if (plus or minus) and len(held) == 1:
            state['busy'] = True
            threading.Thread(target=move, args=('xyz'.index(held[0]),
                                                1.0 if plus else -1.0),
                             daemon=True).start()

    base.schedule_interval(tick, interval=TICK)
    try:
        base.run()
    finally:
        cobotta.close()                 # motors off, arm given back
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--robot', choices=('sampler', 'picker'),
                        default='sampler')
    parser.add_argument('--step', type=float, default=1.0,
                        help='step in mm (default 1)')
    args = parser.parse_args()
    sys.exit(0 if main(args.robot, float(np.clip(args.step / 1000, MIN_STEP,
                                                 MAX_STEP))) else 1)
