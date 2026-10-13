"""Drive the REAL XH430 four-bar cutter -- the hardware twin of ``xh430_cutter.py``.

One Dynamixel XH430 (RS485, Protocol 2.0) turns link1; link2 / link3 follow
through the four-bar, so ONE motor angle is the whole cutter state. On the
servo's own dial the blade is RETRACTED at 180 deg and FULLY DEPLOYED at
337 deg (180 + the sim model's ``TRAVEL``). Everything above is expressed as
the link1 joint angle ``q`` of the sim model, ``q = servo_deg - 180``, so
0 deg is retracted, 157 deg deployed, and ``cutter.fk(qs=[np.radians(q)])``
draws the same pose on screen (``cutter.deploy()`` / ``retract()`` for the
two ends).

Current-based position control (operating mode 5), like the gripper: if the
blade jams on something it stalls at the current ceiling instead of driving
into an overload shutdown.

The blade is ultrasonic, switched by the relay on the CONTEC DIO board
(``coripps.drivers.contec_dio``), which also drives the vacuum generator that
holds the leaf. :meth:`XH430CutterExec.ultrasonic` and
:meth:`XH430CutterExec.vacuum` pass those through, so one object reaches the
whole end effector -- but every verb does ONE thing: ``deploy`` / ``retract``
only move, and the sequencing (when to suck, cut, blow) is the caller's. The
board belongs to the cutter alone: it is opened with the object and closed
with it.

This module is dedicated to THIS cutter -- every parameter lives in the frozen
:class:`Config` below, not in a config file:

* The X series encoder is absolute within one turn, so there is no homing:
  180 deg is encoder tick 2048 (the dot on the flange), 337 deg is 3834.
* In mode 5 the servo does NOT enforce its own position limits, so the
  ``0..travel_deg`` clamp in :meth:`XH430CutterExec.move_to_deg` is the ONLY
  guard on the range -- do not bypass it.

Usage::

    py -3.12 coripps/end_effectors/xh430_cutter/exec.py status
    py -3.12 coripps/end_effectors/xh430_cutter/exec.py deploy
    py -3.12 coripps/end_effectors/xh430_cutter/exec.py retract
    py -3.12 coripps/end_effectors/xh430_cutter/exec.py angle 75     # link1 degrees
    py -3.12 coripps/end_effectors/xh430_cutter/exec.py cycle -n 5
    py -3.12 coripps/end_effectors/xh430_cutter/exec.py release      # torque off
    py -3.12 coripps/end_effectors/xh430_cutter/exec.py ports        # list adapters

The port is found by the USB adapter's serial number (``usb_serial``), so the
same code works whatever COM number a given PC assigns; ``ports`` shows the
serial number of every adapter.

From other code -- one leaf, robot motions elided::

    from coripps.end_effectors.xh430_cutter.exec import XH430CutterExec
    with XH430CutterExec() as cutter:
        # ... robot to the leaf
        cutter.vacuum('suck')       # hold the leaf
        # ... lift slightly
        cutter.ultrasonic(True)
        cutter.deploy()             # the cut happens on the swing out
        cutter.ultrasonic(False)
        cutter.retract()
        # ... robot to the tube
        cutter.vacuum('blow')       # blow the leaf off
        cutter.vacuum('off')
"""

import sys
import time
import argparse
from dataclasses import dataclass

import numpy as np

from one.drivers.dynamixel import (Dynamixel, DynamixelError,
                                   MODE_CURRENT_POSITION, serial_ports)
from coripps.drivers.contec_dio import ContecDIO, ContecDIOError
import coripps.end_effectors.xh430_cutter.xh430_cutter as xh430c


# ---- fixed configuration for THIS cutter ----------------------------------
@dataclass(frozen=True)
class Config:
    """Everything that describes THIS piece of hardware, in one block.

    Frozen and attribute-accessed on purpose: a typo is caught where it is
    written instead of surfacing as a KeyError mid-motion, and the numbers
    cannot be reassigned by accident at runtime.
    """

    # The FTDI adapter's own serial number -- the same on every PC, unlike
    # the COM number, which is looked up from it at connect time.
    usb_serial: str = 'BG03R925A'
    baud: int = 57600
    dxl_id: int = 1
    dio_device: str = 'DIO000'  # CONTEC board: ultrasonic relay + vacuum

    # Servo-dial angle (deg, 0..360 over one turn) with the blade retracted,
    # i.e. q = 0. Deployed is this + travel_deg.
    retract_servo_deg: float = 180.0
    # link1 travel, taken FROM the sim model so the two cannot drift apart
    travel_deg: float = float(np.degrees(xh430c.TRAVEL))

    # Current ceiling while moving. 2.69 mA per unit, so 100 units is ~270 mA.
    # The blade is ultrasonic -- the servo only swings it, never forces a cut.
    # Measured: ~90 mA moving, ~0 holding, and it still ran at 50 units, so
    # this is ~3x headroom (friction, arm pose) and still stalls gently.
    goal_current: int = 100
    profile_velocity: int = 60  # 0.229 rev/min per unit; ~14 rpm
    profile_acceleration: int = 20
    move_timeout: float = 7.0   # a full 0..157 move takes ~4.5 s


CONFIG = Config()

# Tuning of the settle detector -- about the ALGORITHM, not about the cutter,
# so it stays out of Config.
_SETTLE_TICKS = 2               # movement below this counts as stopped
_SETTLE_POLLS = 4


class XH430CutterExec:
    """The XH430 cutter as deploy/retract verbs instead of registers."""

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.dxl = Dynamixel(usb_serial=cfg.usb_serial, baudrate=cfg.baud,
                             dxl_id=cfg.dxl_id)
        try:
            self.dio = ContecDIO(cfg.dio_device)
        except OSError as exc:                  # cdio.dll not installed
            raise ContecDIOError(
                f'cannot load the CONTEC driver: {exc}') from exc

    # ---- connection -------------------------------------------------------
    def connect(self, setup=True, check_error=True):
        """``check_error=False`` skips refusing on a latched hardware error
        -- only for ``clear``, which exists to reboot out of one."""
        self.dxl.connect()
        model, firmware = self.dxl.ping()
        print(f'cutter: {self.dxl.model_name(model)} id {self.dxl.dxl_id} '
              f'on {self.dxl.port} (firmware {firmware})')
        bits, desc = self.dxl.hardware_error()
        if bits and check_error:
            raise DynamixelError(
                f'servo reports a latched hardware error: {desc} -- power '
                f'cycle it or run "clear" to reboot it')
        if setup:
            self.setup()
        return self

    def close(self):
        """Close the board -- which switches EVERY output off, so the blade
        stops and the vacuum drops a held leaf -- and the servo port."""
        try:
            self.dio.close()
        finally:
            self.dxl.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def setup(self):
        """Current-based position control, a gentle profile, torque ON.

        Idempotent: torque is only cycled when the (EEPROM) operating mode
        actually has to change, so calling it again mid-session is harmless.
        """
        if self.dxl.read('operating_mode') != MODE_CURRENT_POSITION:
            self.dxl.set_operating_mode(MODE_CURRENT_POSITION)
            print('cutter: switched to current-based position control')
        # profile_velocity 0 means FULL SPEED (factory default), so it is
        # always written before the first move.
        self.dxl.write('profile_velocity', self.cfg.profile_velocity)
        self.dxl.write('profile_acceleration', self.cfg.profile_acceleration)
        if not self.dxl.read('torque_enable'):
            self.dxl.torque(True)
        self.dxl.write('goal_current', self.cfg.goal_current)

    def release(self):
        """Torque off -- link1 goes limp."""
        self.dxl.torque(False)
        print('cutter: torque off')

    # ---- DIO: ultrasonic blade + vacuum ------------------------------------
    def ultrasonic(self, on):
        """Switch the ultrasonic blade on / off."""
        self.dio.ultrasonic_cutter(on)
        print(f'cutter: ultrasonic {"on" if on else "off"}')

    def vacuum(self, mode):
        """Vacuum generator: ``'suck'`` holds the leaf, ``'blow'`` blows it
        off, ``'off'`` stops both (the driver interlocks suck / blow)."""
        self.dio.vacuum_generator(mode)
        print(f'cutter: vacuum {mode}')

    # ---- state ------------------------------------------------------------
    def servo_deg(self):
        """Present angle on the servo's own dial (180 retracted, 337 out)."""
        return self.dxl.ticks_to_deg(self.dxl.read('present_position'))

    def angle_deg(self):
        """Present link1 angle in degrees -- the sim model's ``q``."""
        return self.servo_deg() - self.cfg.retract_servo_deg

    def status(self):
        line = (f'angle {self.angle_deg():7.2f} deg  '
                f'servo {self.servo_deg():7.2f} deg  '
                f'encoder {self.dxl.read("present_position")}  '
                f'torque {"on" if self.dxl.read("torque_enable") else "off"}  '
                f'current {self.dxl.present_current_ma():6.1f} mA  '
                f'{self.dxl.read("present_input_voltage") * 0.1:.1f} V  '
                f'{self.dxl.read("present_temperature")} C')
        print(line)
        return line

    # ---- motion -----------------------------------------------------------
    def move_to_deg(self, deg, wait=True):
        """Drive link1 to ``deg`` -- 0 is retracted, ``travel_deg`` deployed.

        The target is clamped to 0..travel_deg; in mode 5 nothing in the
        servo enforces that range, so this clamp is the only guard.
        """
        travel = self.cfg.travel_deg
        target = float(np.clip(deg, 0.0, travel))
        if target != deg:
            print(f'cutter: {deg:.2f} deg clamped to {target:.2f} '
                  f'(travel 0..{travel:g})')
        ticks = self.dxl.deg_to_ticks(self.cfg.retract_servo_deg + target)
        self.dxl.write('goal_current', self.cfg.goal_current)
        self.dxl.write('goal_position', int(ticks))
        if wait:
            self._settle()
        return target

    def _settle(self, timeout=None):
        """Block until the position goes quiet; return ``(ticks, mA)``.

        Position, not the ``moving`` flag: a blade stalled on something never
        reaches its goal, so ``moving`` would stay set until the timeout.
        """
        last = self.dxl.read('present_position')
        quiet = 0
        deadline = time.time() + (self.cfg.move_timeout if timeout is None
                                  else timeout)
        while time.time() < deadline:
            time.sleep(0.05)
            pos = self.dxl.read('present_position')
            quiet = quiet + 1 if abs(pos - last) <= _SETTLE_TICKS else 0
            last = pos
            if quiet >= _SETTLE_POLLS:
                break
        return last, self.dxl.present_current_ma()

    def deploy(self, wait=True):
        """Swing the blade fully out. Motion only -- the ultrasonic blade is
        switched separately with :meth:`ultrasonic`."""
        return self.move_to_deg(self.cfg.travel_deg, wait=wait)

    def retract(self, wait=True):
        """Pull the blade back in."""
        return self.move_to_deg(0.0, wait=wait)


# ---------------------------------------------------------------------------
def _cmd_status(cutter, args):
    cutter.status()


def _cmd_deploy(cutter, args):
    cutter.deploy()
    cutter.status()


def _cmd_retract(cutter, args):
    cutter.retract()
    cutter.status()


def _cmd_angle(cutter, args):
    cutter.move_to_deg(args.degrees)
    cutter.status()


def _cmd_cycle(cutter, args):
    for i in range(args.n):
        cutter.deploy()
        out = (cutter.angle_deg(), cutter.dxl.present_current_ma())
        time.sleep(args.dwell)
        cutter.retract()
        back = (cutter.angle_deg(), cutter.dxl.present_current_ma())
        time.sleep(args.dwell)
        print(f'cycle {i + 1}/{args.n}  '
              f'deployed {out[0]:6.2f} deg @ {out[1]:4.0f} mA   ->   '
              f'retracted {back[0]:6.2f} deg @ {back[1]:4.0f} mA')
    cutter.status()


def _cmd_release(cutter, args):
    cutter.release()


def _cmd_clear(cutter, args):
    bits, desc = cutter.dxl.hardware_error()
    print(f'hardware error 0x{bits:02x}: {desc}')
    cutter.dxl.clear_hardware_error()
    print('cutter: rebooted')


def _cmd_ports(args):
    for device, sn, desc in serial_ports():
        mark = '  <- cutter' if sn == CONFIG.usb_serial else ''
        print(f'{device:6s}  {sn or "-":12s}  {desc}{mark}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)

    def add(name, help_text, **defaults):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(**defaults)
        return p

    add('status', 'read angle / current', func=_cmd_status, setup=False)
    add('deploy', f'swing the blade out to {CONFIG.travel_deg:g} deg',
        func=_cmd_deploy)
    add('retract', 'pull the blade back to 0 deg', func=_cmd_retract)

    p = add('angle', 'go to a link1 angle in degrees', func=_cmd_angle)
    p.add_argument('degrees', type=float)

    p = add('cycle', 'deploy/retract repeatedly', func=_cmd_cycle)
    p.add_argument('-n', type=int, default=3)
    p.add_argument('--dwell', type=float, default=0.5)

    add('release', 'torque off', func=_cmd_release, setup=False)
    add('clear', 'report and reboot out of a hardware error', func=_cmd_clear,
        setup=False, check_error=False)
    add('ports', 'list serial ports and their USB serial numbers',
        func=_cmd_ports, offline=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, 'offline', False):         # needs no servo
        args.func(args)
        return 0
    try:
        cutter = XH430CutterExec()
    except ContecDIOError as exc:
        print(f'cutter: {exc}')
        return 1
    try:
        cutter.connect(setup=getattr(args, 'setup', True),
                       check_error=getattr(args, 'check_error', True))
        args.func(cutter, args)
    except (DynamixelError, ContecDIOError) as exc:
        print(f'cutter: {exc}')
        return 1
    finally:
        cutter.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
