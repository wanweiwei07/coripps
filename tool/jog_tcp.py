"""Jog a robot's tcp, in the viewer: from a panel, or by key chords.

The panel (top left of the viewer window):

    robot             the robot jogged: picker / sampler. Switching gives
                      the one jogged so far back (motors off) and connects
                      to the other; when it cannot be reached, the one
                      before is taken again
    tool / world      jog along the tcp's OWN axes, or the world's
    X- X+ ...  Rz- Rz+  one step along x / y / z, one turn about z (the
                        tcp's own z, or the world's vertical through the
                        tcp), the tcp staying where it is
    align             turn the tcp, on the spot, to the nearest
                      orientation with its z straight down and its x / y
                      along the world's axes
    gripper / cutter  the jogged robot's tool: the picker's gripper open /
                      close, the sampler's cutter deploy / retract (the
                      blade swung out / in -- the ultrasonic stays off);
                      connected at its first press
    step / turn       how far one press goes
    speed             the arm's speed for the jog moves (%)
    pose / joints     where the tcp is, live
    copy pose         the tcp pose as ``pos = (...)`` / ``rotmat = ...``
                      lines to the clipboard (and the terminal), ready to
                      paste into a layout file

The keys still work (the viewer window has to have the focus):

    hold x / y / z, tap + or -   one step along x / y / z
                                 (+: the '=/+' key or keypad +;
                                  -: the '-' key or keypad -)
    hold t, tap + or -           one turn about z
    [ / ]                        halve / double the step and the turn
    p                            print the tcp pose
    close the window             quit

One step per press; presses are ignored while the arm moves. The viewer
shows the cell with both robots at their real joints (the one not jogged:
as last read) and both robots' tcp frames.

The sampler jogs its cutter's 'tcp', the picker its gripper's
'grasp_center'. Each move is a straight line (IK nearest the last, every
SAMPLE apart; a turn every TURN_SAMPLE); nothing moves when there is no
smooth IK for it. NO collision check: a debugging tool, the one at the
controls watches the arm. The arm is enabled at the first move and given
back when the window closes.

    py -3.12 coripps/tool/jog_tcp.py                 # the picker first, 1 mm
    py -3.12 coripps/tool/jog_tcp.py --robot sampler --step 5
"""

import sys
import argparse
import threading

import numpy as np

from one.drivers.cobotta import BCapError, CobottaMotion
import one.utils.math as oum
import one.utils.constant as ouc

from coripps.drivers.cobotta_read import read_qs
from coripps.tool import pose_sequence as pseq


SPEED = 10                      # %, to begin with
MAX_SPEED = 30                  # %, the panel's slider goes up to this
SAMPLE = 0.001                  # m between a move's IK samples
MIN_STEP, MAX_STEP = 0.0001, 0.020      # m
STEPS_MM = (0.1, 0.5, 1, 5, 10, 20)     # the panel's step choices
TURN_SAMPLE = np.radians(1.0)   # between a turn's IK samples
TURN_STEP = np.radians(5.0)     # the turn per press, to begin with
MIN_TURN, MAX_TURN = np.radians(0.5), np.radians(45.0)
TURNS_DEG = (0.5, 1, 5, 15, 45)         # the panel's turn choices
TICK = 1 / 30.0                 # s
FRAME_LENGTH = 0.05             # m, the drawn tcp frame
_SAMPLER_HOST = '192.168.0.101'
_PICKER_HOST = '192.168.0.102'


# the robots that can be jogged: (host, its tool, the tool's tcp jogged)
ROBOTS = {'picker': (_PICKER_HOST, 'gripper', 'grasp_center'),
          'sampler': (_SAMPLER_HOST, 'cutter', 'tcp')}


# each robot's tool, from the panel: (its name, the buttons, as
# (label, the exec's verb))
TOOL_VERBS = {'picker': ('gripper', (('open', 'open'), ('close', 'close'))),
              'sampler': ('cutter', (('deploy', 'deploy'),
                                     ('retract', 'retract')))}


def tool_exec(robot):
    """The real tool of ``robot``, connected: the picker's gripper, the
    sampler's cutter (with its DIO board)."""
    if robot == 'picker':
        from coripps.end_effectors.xh430_gripper.exec import XH430GripperExec
        return XH430GripperExec().connect()
    from coripps.end_effectors.xh430_cutter.exec import XH430CutterExec
    return XH430CutterExec().connect()


def past_limits(mech, qs):
    """The joints of ``qs`` beyond ``mech``'s model limits, as text ('' when
    none). Kept as read, NOT clamped: the real arm can stand a little past
    a model limit (the sampler's J4 at -171.9 against -170), and clamping
    would show -- and solve the IK from -- a pose it is not in."""
    c = mech.structure.compiled
    lo, hi = np.degrees(c.jlmt_low_by_idx), np.degrees(c.jlmt_high_by_idx)
    q = np.degrees(qs)
    return ', '.join(
        f'J{i + 1} {q[i]:.1f} (limit {lo[i] if q[i] < lo[i] else hi[i]:g})'
        for i in range(len(q)) if not lo[i] <= q[i] <= hi[i])


def tcp_tf(mech, tcp, qs):
    mech.fk(qs=qs)
    return np.array(tcp.tf, dtype=np.float64)


def pose_text(tf, qs):
    """The tcp pose, for the terminal."""
    return (f'tcp {np.round(tf[:3, 3] * 1000, 2)} mm  x '
            f'{np.round(tf[:3, 0], 3)}  y {np.round(tf[:3, 1], 3)}  z '
            f'{np.round(tf[:3, 2], 3)}  joints {np.round(np.degrees(qs), 2)} '
            'deg')


def pose_code(tf, name):
    """The tcp pose as Python lines to paste into a layout file."""
    pos = ', '.join(f'{v:.4f}' for v in tf[:3, 3])
    rows = ',\n                   '.join(
        '[' + ', '.join(f'{v:+.4f}' for v in row) + ']' for row in tf[:3, :3])
    return (f'pos = ({pos})          # m, {name} in the world\n'
            f'rotmat = np.array([{rows}])')


def jog_path(mech, tcp, qs, axis, dist, world=False):
    """``(path, why)``: a straight path moving the tcp ``dist`` along its own
    ``axis`` (0 / 1 / 2) -- ``world``: the world's -- ``qs`` first; path None
    and ``why`` it cannot be."""
    tf = tcp_tf(mech, tcp, qs)
    delta = dist * (np.eye(3)[axis] if world else tf[:3, axis])
    n = max(int(np.ceil(abs(dist) / SAMPLE)), 1)
    path = pseq.follow_positions(
        mech, qs, [(tf[:3, 3] + delta * k / n).astype(np.float32)
                   for k in range(1, n + 1)],
        tf[:3, :3].astype(np.float32), tcp=tcp)
    if path is None:
        return None, 'no smooth IK for that move -- not moved'
    return path, ''


def turn_path(mech, tcp, qs, angle, world=False):
    """``(path, why)``: a path turning the tcp ``angle`` (rad) about its OWN
    z -- ``world``: the world's vertical through it -- its position kept,
    ``qs`` first; path None and ``why`` it cannot be."""
    tf = tcp_tf(mech, tcp, qs)
    pos = tf[:3, 3].astype(np.float32)
    n = max(int(np.ceil(abs(angle) / TURN_SAMPLE)), 1)
    path = [np.asarray(qs, dtype=np.float32)]
    for k in range(1, n + 1):
        turn = oum.rotmat_from_axangle(ouc.StandardAxis.Z, angle * k / n)
        rot = turn @ tf[:3, :3] if world else tf[:3, :3] @ turn
        step = pseq.follow_positions(mech, path[-1], [pos],
                                     rot.astype(np.float32), tcp=tcp)
        if step is None:
            return None, 'no smooth IK for that turn -- not turned'
        path.append(step[-1])
    return np.asarray(path, dtype=np.float32), ''


# pointing straight down, x along world -x: Rz(k * 90 deg) of it are the
# four orientations "z down, x / y along the world's axes"
_DOWN = oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi)
_AXES = ('-x', '-y', '+x', '+y')        # where the tcp's x points, by k


def nearest_aligned(rotmat):
    """``(rotmat, angle, x_dir)``: the orientation with z straight down and
    x / y along the world's axes nearest ``rotmat``, the turn to it (rad)
    and where its x points ('+x' / '-y' / ...)."""
    best = None
    for k in range(4):
        cand = oum.rotmat_from_axangle(ouc.StandardAxis.Z,
                                       k * np.pi / 2) @ _DOWN
        _, angle = oum.axangle_between_rotmat(rotmat, cand)
        if best is None or angle < best[1]:
            best = (cand, float(angle), _AXES[k])
    return best


def align_path(mech, tcp, qs):
    """``(path, why, angle, x_dir)``: a path turning the tcp, on the spot,
    to :func:`nearest_aligned` (slerp, TURN_SAMPLE apart), ``qs`` first;
    path None and ``why`` it cannot be (or there is nothing to do)."""
    tf = tcp_tf(mech, tcp, qs)
    pos = tf[:3, 3].astype(np.float32)
    target, angle, x_dir = nearest_aligned(tf[:3, :3])
    if angle < np.radians(0.01):
        return None, 'already aligned', angle, x_dir
    n = max(int(np.ceil(angle / TURN_SAMPLE)), 1)
    path = [np.asarray(qs, dtype=np.float32)]
    for rot in oum.rotmat_slerp(tf[:3, :3], target, n + 1)[1:]:
        step = pseq.follow_positions(mech, path[-1], [pos], rot, tcp=tcp)
        if step is None:
            why = 'no smooth IK for that turn -- not turned'
            return None, why, angle, x_dir
        path.append(step[-1])
    return np.asarray(path, dtype=np.float32), '', angle, x_dir


def main(robot='picker', step=0.001):
    from pyglet.window import key
    from imgui_bundle import imgui
    import one.viewer.world as ovw
    import one.viewer.imgui_overlay as ovio
    from coripps.robots.mobisys import mobisys

    # two copies of the cell: one for the IK (worked on from the move
    # thread), one on screen (posed only from the viewer's tick)
    _, calc_picker, calc_sampler = mobisys()
    mobisys_base, picker, sampler = mobisys()
    calc = {'picker': calc_picker, 'sampler': calc_sampler}
    seen = {'picker': picker, 'sampler': sampler}
    arms = {}
    for name, (host, tool, tcp_name) in ROBOTS.items():
        mech = calc[name]
        arms[name] = {'host': host, 'mech': mech, 'tcp_name': tcp_name,
                      'tcp': getattr(mech, tool).tcp(tcp_name),
                      'shown': seen[name], 'tool': getattr(seen[name], tool),
                      'qs': None}
    state = {'robot': None, 'motion': None, 'step': step, 'turn': TURN_STEP,
             'busy': False, 'world': False, 'speed': SPEED, 'msg': 'ready',
             'bad': False, 'tools': {}, 'tool_view': {}}

    def say(msg, bad=False):
        state['msg'], state['bad'] = msg, bad
        print(f'  {msg}')

    def take(name):
        """Connect to ``name`` and make it the jogged robot; the one jogged
        so far is given back first (motors off). False (said why) when it
        cannot be reached -- nothing jogged then."""
        old = state['robot']
        if state['motion'] is not None:
            mech = arms[old]['mech']
            arms[old]['qs'] = np.float32(state['motion'].joint_angles(
                ndof=mech.ndof))
            state['motion'].disconnect()            # motors off, arm back
            state['motion'], state['robot'] = None, None
        arm = arms[name]
        try:
            motion = CobottaMotion(arm['host'], speed=state['speed']).connect()
            arm['qs'] = np.float32(motion.joint_angles(
                ndof=arm['mech'].ndof))
        except (OSError, BCapError) as exc:
            say(f"{name}: cannot reach {arm['host']} ({exc})", bad=True)
            return False
        state['motion'], state['robot'] = motion, name
        say(f"jogging the {name}'s {arm['tcp_name']}")
        print(pose_text(tcp_tf(arm['mech'], arm['tcp'], arm['qs']),
                        arm['qs']))
        return True

    def switch(name):
        old = state['robot']
        try:
            if not take(name) and old is not None:
                msg = state['msg']
                if take(old):                       # back to the one before
                    say(f'{msg} -- still jogging the {old}', bad=True)
        except (OSError, BCapError) as exc:
            say(f'switching to the {name}: {exc}', bad=True)
        finally:
            state['busy'] = False

    # the other robot (not jogged to begin with) at its real joints
    for name, arm in arms.items():
        if name != robot:
            qs = read_qs(arm['host'], arm['mech'].ndof)
            if qs is not None:
                arm['qs'] = np.float32(qs)
    if not take(robot):
        return False

    first = arms[robot]
    centre = tcp_tf(first['mech'], first['tcp'], first['qs'])[:3, 3]
    base = ovw.World(cam_pos=centre + np.array([0.6, 0.45, 0.35]),
                     cam_lookat_pos=centre)
    mobisys_base.attach_to(base.scene)
    picker.attach_to(base.scene)
    picker.gripper.attach_to(base.scene)
    sampler.attach_to(base.scene)
    scale = FRAME_LENGTH / 0.2          # the frame primitive is 0.2 m long
    for arm in arms.values():           # both robots' tcp frames
        arm['tool'].toggle_tcp(arm['tcp_name'], length_scale=scale,
                               radius_scale=scale)

    def move(axis, sign):
        """``axis`` 0 / 1 / 2: along x / y / z; 3: turn about z; 4: align
        (z down, x / y along the world's axes; ``sign`` unused)."""
        name, world = state['robot'], state['world']
        arm, motion = arms[name], state['motion']
        mech, tcp = arm['mech'], arm['tcp']
        try:
            if axis == 4:
                path, why, angle, x_dir = align_path(mech, tcp, arm['qs'])
                label = (f'{name} align: z down, x along {x_dir} '
                         f'({np.degrees(angle):.1f} deg)')
            elif axis == 3:
                path, why = turn_path(mech, tcp, arm['qs'],
                                      sign * state['turn'], world)
                what = f"{np.degrees(state['turn']):g} deg"
            else:
                path, why = jog_path(mech, tcp, arm['qs'], axis,
                                     sign * state['step'], world)
                what = f"{state['step'] * 1000:g} mm"
            if axis != 4:
                label = (f"{name} {'world ' if world else ''}"
                         f"{('x', 'y', 'z', 'rz')[axis]}"
                         f"{'+' if sign > 0 else '-'} {what}")
            if path is None:
                say(f'{label}: {why}', bad=why != 'already aligned')
            else:
                motion.enable(speed=state['speed'])
                motion.set_speed(state['speed'])
                motion.move_path(path[1:])
            arm['qs'] = np.float32(motion.joint_angles(ndof=mech.ndof))
            if path is not None:
                say(label)
                print(pose_text(tcp_tf(mech, tcp, arm['qs']), arm['qs']))
        except (OSError, BCapError) as exc:
            say(f'{name}: {exc}', bad=True)
        finally:
            state['busy'] = False

    def start(target, *args, msg='moving ...'):
        if state['busy']:
            return
        if target is move and state['robot'] is None:
            say('no robot connected -- pick one', bad=True)
            return
        state['busy'] = True
        state['msg'], state['bad'] = msg, False
        threading.Thread(target=target, args=args, daemon=True).start()

    def use_tool(name, verb):
        """``verb`` (open / close, deploy / retract) on ``name``'s tool,
        connected at its first use; what it is at afterwards goes to the
        view."""
        tool_name = TOOL_VERBS[name][0]
        try:
            if name not in state['tools']:
                state['tools'][name] = tool_exec(name)
            ex = state['tools'][name]
            getattr(ex, verb)()
            if name == 'picker':
                mm = ex.opening_mm()
                state['tool_view'][name] = mm / 1000
                say(f'{name} {tool_name} {verb}: {mm:.1f} mm')
            else:
                deg = ex.angle_deg()
                state['tool_view'][name] = np.radians(deg)
                say(f'{name} {tool_name} {verb}: {deg:.1f} deg')
        except Exception as exc:        # serial / DIO / servo errors alike
            say(f'{name} {tool_name} {verb}: {exc}', bad=True)
        finally:
            state['busy'] = False

    def panel():
        w = imgui.get_frame_height() * 2.6         # a jog button's width
        imgui.set_next_window_pos((20, 20), imgui.Cond_.first_use_ever)
        imgui.begin('jog tcp', None, imgui.WindowFlags_.always_auto_resize)
        # which robot
        imgui.begin_disabled(state['busy'])
        imgui.text('robot')
        for name in ROBOTS:
            imgui.same_line()
            if (imgui.radio_button(name, state['robot'] == name)
                    and state['robot'] != name):
                start(switch, name, msg=f'connecting to the {name} ...')
        imgui.end_disabled()
        name = state['robot']
        if name is not None:
            imgui.same_line()
            imgui.text_disabled(f"({arms[name]['tcp_name']})")
        imgui.separator()
        # jog
        for i, frame in enumerate(('tool', 'world')):
            if i:
                imgui.same_line()
            if imgui.radio_button(frame, state['world'] == bool(i)):
                state['world'] = bool(i)
        imgui.begin_disabled(state['busy'] or name is None)
        for axis, label in enumerate(('X', 'Y', 'Z', 'Rz')):
            if imgui.button(f'{label}-', (w, 0)):
                start(move, axis, -1.0)
            imgui.same_line()
            if imgui.button(f'{label}+', (w, 0)):
                start(move, axis, 1.0)
        if imgui.button('align: z down, xy || world', (2 * w + imgui.get_style(
                ).item_spacing.x, 0)):
            start(move, 4, 1.0)
        if name is not None:
            tool_name, verbs = TOOL_VERBS[name]
            imgui.text(f'{tool_name:8s}')
            for label, verb in verbs:
                imgui.same_line()
                if imgui.button(f'{label}##tool', (w, 0)):
                    start(use_tool, name, verb,
                          msg=f'{tool_name} {label} ...')
        imgui.end_disabled()
        imgui.separator()
        # step / turn / speed
        imgui.text('step mm')
        for mm in STEPS_MM:
            imgui.same_line()
            if imgui.radio_button(f'{mm:g}##s', bool(np.isclose(
                    state['step'] * 1000, mm))):
                state['step'] = mm / 1000
        imgui.text('turn deg')
        for deg in TURNS_DEG:
            imgui.same_line()
            if imgui.radio_button(f'{deg:g}##t', bool(np.isclose(
                    np.degrees(state['turn']), deg))):
                state['turn'] = np.radians(deg)
        _, state['speed'] = imgui.slider_int('speed %', state['speed'], 1,
                                             MAX_SPEED)
        imgui.separator()
        # where it is
        if name is not None:
            arm = arms[name]
            tf = tcp_tf(arm['mech'], arm['tcp'], arm['qs'])
            p = tf[:3, 3] * 1000
            imgui.text(f'pos  {p[0]:8.2f} {p[1]:8.2f} {p[2]:8.2f} mm')
            for i, ax in enumerate('xyz'):
                a = tf[:3, i]
                imgui.text(f'{ax} axis {a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}')
            imgui.text('joints ' + ' '.join(
                f'{v:.1f}' for v in np.degrees(arm['qs'])))
            past = past_limits(arm['mech'], arm['qs'])
            if past:
                imgui.text_colored((1.0, 0.35, 0.3, 1.0),
                                   f'past the model limit: {past}')
            if imgui.button('copy pose'):
                code = pose_code(tf, f"the {name}'s {arm['tcp_name']}")
                imgui.set_clipboard_text(code)
                print(code)
                say('pose copied to the clipboard')
            imgui.separator()
        if state['bad']:
            imgui.text_colored((1.0, 0.35, 0.3, 1.0), state['msg'])
        else:
            imgui.text(state['msg'])
        imgui.end()

    ovio.ImGuiOverlay(base, panel)

    axis_keys = {'x': key.X, 'y': key.Y, 'z': key.Z, 't': key.T}
    plus_keys = (key.EQUAL, key.PLUS, key.NUM_ADD)
    minus_keys = (key.MINUS, key.NUM_SUBTRACT)

    def tick(dt):
        im = base.input_manager
        # read every edge each tick, so none is left over for later
        plus = any([im.is_key_pressed_edge(k) for k in plus_keys])
        minus = any([im.is_key_pressed_edge(k) for k in minus_keys])
        smaller = im.is_key_pressed_edge(key.BRACKETLEFT)
        larger = im.is_key_pressed_edge(key.BRACKETRIGHT)
        show = im.is_key_pressed_edge(key.P)
        for arm in arms.values():
            if arm['qs'] is not None:
                arm['shown'].fk(qs=arm['qs'])
        # a tool's real state, read after a press, on the model shown
        # (the scene is changed here, in the viewer's thread)
        for name, value in list(state['tool_view'].items()):
            del state['tool_view'][name]
            tool = arms[name]['tool']
            if name == 'picker':
                lo, hi = tool.jaw_range
                tool.set_opening(float(np.clip(value, lo, hi)))
            else:
                tool.fk(qs=[value])
        if state['busy']:
            return
        if smaller or larger:
            k = 0.5 if smaller else 2.0
            state['step'] = float(np.clip(state['step'] * k, MIN_STEP,
                                          MAX_STEP))
            state['turn'] = float(np.clip(state['turn'] * k, MIN_TURN,
                                          MAX_TURN))
            print(f'step {state["step"] * 1000:g} mm, turn '
                  f'{np.degrees(state["turn"]):g} deg')
        if show and state['robot'] is not None:
            arm = arms[state['robot']]
            print(pose_text(tcp_tf(arm['mech'], arm['tcp'], arm['qs']),
                            arm['qs']))
        held = [a for a, k in axis_keys.items() if im.is_key_pressed(k)]
        if (plus or minus) and len(held) == 1:
            start(move, 'xyzt'.index(held[0]), 1.0 if plus else -1.0)

    base.schedule_interval(tick, interval=TICK)
    try:
        base.run()
    finally:
        if state['motion'] is not None:
            state['motion'].disconnect()    # motors off, arm given back
        for ex in state['tools'].values():
            ex.disconnect()     # the gripper keeps its grip, the cutter's
                                # DIO goes off (blade, vacuum)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--robot', choices=('sampler', 'picker'),
                        default='picker')
    parser.add_argument('--step', type=float, default=1.0,
                        help='step in mm (default 1)')
    args = parser.parse_args()
    sys.exit(0 if main(args.robot, float(np.clip(args.step / 1000, MIN_STEP,
                                                 MAX_STEP))) else 1)
