"""The XH430 four-bar cutter, simulated and then executed on the real hardware.

The viewer loops the retract/deploy motion on the SIM cutter. Press ``g`` and
the same motion is sent to the real servo -- and while it runs, the sim is
driven from the servo's own encoder, so what you see on screen is the real
link1 position rather than a replay of the animation.

Both halves come from ``coripps.end_effectors.xh430_cutter``:
``xh430_cutter`` for the four-bar, ``exec`` for the hardware. The angles are
the same numbers on both sides (link1 ``q``: 0 deg retracted, ``TRAVEL``
deployed), which is the point of the demo -- if the screen and the bench
disagree, the calibration is wrong.

Running with no servo attached is fine: the preview keeps looping and ``g``
just says there is nothing to drive.

Keys:  g = run it for real    r = torque off    esc = quit

    py -3.12 coripps/example/xh430cutter_sim_exec.py
"""

import time

import numpy as np
import pyglet.window.key as pkey

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

import coripps.end_effectors.xh430_cutter.xh430_cutter as xh430c
from coripps.end_effectors.xh430_cutter.exec import (XH430CutterExec,
                                                     DynamixelError,
                                                     ContecDIOError)


_TICK = 0.03                     # viewer tick, seconds
_SWEEP_SECONDS = 4.0             # one retracted -> deployed -> retracted sweep
_DWELL = 1.0                     # pause at each end while executing
_SETTLE_TICKS = 2                # encoder movement below this counts as stopped
_SETTLE_POLLS = 4
_EXEC_TIMEOUT = 7.0              # give up waiting on one move (a full one ~4.3 s)


def preview_angle(elapsed, travel_deg):
    """Triangle sweep 0 -> travel -> 0, so the loop has no jump at the seam."""
    phase = (elapsed % _SWEEP_SECONDS) / _SWEEP_SECONDS
    return travel_deg * (2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase))


def main():
    base = ovw.World(cam_pos=[0.0, 0.25, 0.04], cam_lookat_pos=[0, 0, 0.04])
    cutter = xh430c.XH430Cutter()
    cutter.retract()
    cutter.attach_to(base.scene)
    ossop.frame().attach_to(base.scene)

    real = None
    try:
        real = XH430CutterExec()
        real.connect()
    except (DynamixelError, ContecDIOError) as exc:
        if real is not None:
            real.close()            # the board opened even if the servo didn't
        real = None
        print(f'{exc}\n         preview only -- "g" has nothing to drive')

    travel_deg = float(np.degrees(xh430c.TRAVEL))   # exec uses the same
    # (XH430CutterExec verb, dwell) pairs
    program = [('deploy', _DWELL), ('retract', _DWELL)]

    state = {'mode': 'preview', 'elapsed': 0.0, 'step': 0,
             'last': None, 'quiet': 0, 'deadline': 0.0, 'until': 0.0}

    def show_angle(deg):
        """Drive the sim cutter, clipped to the joint it actually has."""
        q = float(np.clip(np.deg2rad(deg), 0.0, xh430c.TRAVEL))
        cutter.fk(qs=[q])
        base.scene.dirty = True

    def issue(step):
        """Start one program step on the real servo."""
        verb, _ = program[step]
        target = getattr(real, verb)(wait=False)
        print(f'  step {step + 1}/{len(program)}: {verb} to {target:.1f} deg')
        state['last'] = None
        state['quiet'] = 0
        state['deadline'] = time.time() + _EXEC_TIMEOUT

    def start_execution():
        if real is None:
            print('no servo on the bus -- nothing to execute')
            return
        print('executing on the real cutter')
        state['mode'] = 'execute'
        state['step'] = 0
        issue(0)

    def step_execution():
        """One encoder read per tick: mirror it, and watch for it to settle.

        Settling is judged by the position going quiet rather than by the
        ``moving`` flag, so a blade stalled on something still moves on.
        """
        pos = real.dxl.read('present_position')
        show_angle(real.dxl.ticks_to_deg(pos) - real.cfg.retract_servo_deg)
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
        print(f'    settled at {real.angle_deg():6.2f} deg @ '
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
        show_angle(preview_angle(state['elapsed'], travel_deg))

    base.schedule_interval(tick, interval=_TICK)
    base.set_caption('XH430 cutter -- g: run for real, r: torque off')
    print(f'preview: sweeping 0..{travel_deg:.1f} deg on the sim')
    print("'g' runs the same motion on the real cutter, 'r' releases it")
    return base, cutter, real, state


if __name__ == '__main__':
    import builtins

    base, cutter, real, state = main()
    builtins.base = base
    builtins.cutter = cutter
    builtins.real = real
    builtins.state = state
    try:
        base.run()
    finally:
        if real is not None:
            real.release()
            real.close()
