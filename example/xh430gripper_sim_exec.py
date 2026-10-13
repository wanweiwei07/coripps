"""The XH430 rotary gripper, simulated and then executed on the real hardware.

The viewer loops the open/close motion on the SIM gripper. Press ``g`` and the
same motion is sent to the real servo on COM5 -- and while it runs, the sim is
driven from the servo's own encoder, so what you see on screen is the real
blade position rather than a replay of the animation.

Both halves come from ``coripps.end_effectors.xh430_gripper``:
``xh430_gripper`` for the geometry, ``exec`` for the hardware. The angles are
the same numbers on both sides (0 deg closed, ``Config.open_deg`` open), which
is the point of the demo -- if the screen and the bench disagree, the
calibration is wrong.

Running with no servo attached is fine: the preview keeps looping and ``g``
just says there is nothing to drive.

Keys:  g = run it for real    r = torque off    esc = quit

    py -3.12 coripps/example/xh430gripper_sim_exec.py
"""

import time

import numpy as np
import pyglet.window.key as pkey

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

import coripps.end_effectors.xh430_gripper.xh430_gripper as xh430g
from coripps.end_effectors.xh430_gripper.exec import (XH430GripperExec,
                                                      DynamixelError)


_TICK = 0.03                     # viewer tick, seconds
_SWEEP_SECONDS = 2.0             # one closed -> open -> closed preview sweep
_DWELL = 1.0                     # pause at each end while executing
_SETTLE_TICKS = 2                # encoder movement below this counts as stopped
_SETTLE_POLLS = 4
_EXEC_TIMEOUT = 4.0              # give up waiting on one move


def preview_angle(elapsed, open_deg):
    """Triangle sweep 0 -> open -> 0, so the loop has no jump at the seam."""
    phase = (elapsed % _SWEEP_SECONDS) / _SWEEP_SECONDS
    return open_deg * (2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase))


def main():
    base = ovw.World(cam_pos=[0.3, 0.22, 0.2], cam_lookat_pos=[0, 0, 0.06])
    gripper = xh430g.XH430Gripper()
    gripper.fk(qs=[0.0])
    gripper.attach_to(base.scene)
    ossop.frame().attach_to(base.scene)
    gc_tf = gripper.tcp('grasp_center').tf
    ossop.frame(pos=gc_tf[:3, 3], rotmat=gc_tf[:3, :3],
                color_mat=ouc.CoordColor.MYC).attach_to(base.scene)

    try:
        real = XH430GripperExec().connect()
    except DynamixelError as exc:
        real = None
        print(f'{exc}\n         preview only -- "g" has nothing to drive')

    open_deg = real.cfg.open_deg if real is not None \
        else float(np.degrees(xh430g.OPEN_TRAVEL))
    # (open, dwell) pairs; None means "close and grip"
    program = [(open_deg, _DWELL), (None, _DWELL)]

    state = {'mode': 'preview', 'elapsed': 0.0, 'step': 0,
             'last': None, 'quiet': 0, 'deadline': 0.0, 'until': 0.0}

    def show_angle(deg):
        """Drive the sim gripper, clipped to the joint it actually has.

        The real gripper is deliberately driven a degree PAST zero to grip, and
        the sim hinge stops at zero -- clipping keeps that overshoot from
        pushing the model out of its limits.
        """
        q = float(np.clip(np.deg2rad(deg), 0.0, xh430g.TRAVEL))
        gripper.fk(qs=[q])
        base.scene.dirty = True

    def issue(step):
        """Start one program step on the real servo."""
        target, _ = program[step]
        if target is None:
            real.close_jaws(wait=False)
            print(f'  step {step + 1}/{len(program)}: closing onto the stop')
        else:
            real.move_to_deg(target, wait=False)
            print(f'  step {step + 1}/{len(program)}: opening to '
                  f'{target:.1f} deg')
        state['last'] = None
        state['quiet'] = 0
        state['deadline'] = time.time() + _EXEC_TIMEOUT

    def start_execution():
        if real is None:
            print('no servo on the bus -- nothing to execute')
            return
        print('executing on the real gripper')
        state['mode'] = 'execute'
        state['step'] = 0
        issue(0)

    def step_execution():
        """One encoder read per tick: mirror it, and watch for the stall.

        Settling is judged by the position going quiet rather than by the
        ``moving`` flag, because a grip never reaches its goal position.
        """
        pos = real.dxl.read('present_position')
        show_angle(real.dxl.ticks_to_deg(pos - real.cfg.zero_ticks) *
                   real.cfg.sign)
        last = state['last']
        state['quiet'] = (state['quiet'] + 1
                          if last is not None and abs(pos - last) <= _SETTLE_TICKS
                          else 0)
        state['last'] = pos
        if state['quiet'] < _SETTLE_POLLS and time.time() < state['deadline']:
            return
        if state['quiet'] < _SETTLE_POLLS:
            print(f'  step {state["step"] + 1}: still moving after '
                  f'{_EXEC_TIMEOUT}s -- moving on')
        if program[state['step']][0] is None:
            real.hold()                 # close_jaws(wait=False) left it to us
        print(f'    settled at {real.angle_deg():5.2f} deg / '
              f'{real.opening_mm():5.2f} mm @ '
              f'{real.dxl.present_current_ma():4.0f} mA')
        state['until'] = time.time() + program[state['step']][1]
        state['mode'] = 'dwell'

    def step_dwell():
        if time.time() < state['until']:
            return
        state['step'] += 1
        if state['step'] >= len(program):
            state['mode'] = 'preview'
            print("done -- back to the preview loop ('g' runs it again, "
                  "'r' releases)")
            return
        state['mode'] = 'execute'
        issue(state['step'])

    def tick(dt):
        if state['mode'] == 'execute':
            step_execution()
            return
        if state['mode'] == 'dwell':
            step_dwell()
            return
        if base.input_manager.is_key_pressed_edge(pkey.G):
            start_execution()
            return
        if base.input_manager.is_key_pressed_edge(pkey.R):
            if real is not None:
                real.release()
            return
        state['elapsed'] += dt
        show_angle(preview_angle(state['elapsed'], open_deg))

    base.schedule_interval(tick, interval=_TICK)
    base.set_caption('XH430 rotary gripper -- g: run for real, r: torque off')
    print(f'preview: sweeping 0..{open_deg:.1f} deg on the sim '
          f"({gripper.jaw_range[0] * 1000:.1f}.."
          f'{gripper.opening_from_angle(np.deg2rad(open_deg)) * 1000:.1f} mm)')
    print("'g' runs the same motion on the real gripper, 'r' releases it")
    return base, gripper, real, state


if __name__ == '__main__':
    import builtins

    base, gripper, real, state = main()
    builtins.base = base
    builtins.gripper = gripper
    builtins.real = real
    builtins.state = state
    try:
        base.run()
    finally:
        if real is not None:
            real.release()
            real.close()
