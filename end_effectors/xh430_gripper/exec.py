"""Drive the REAL XH430 rotary gripper -- the hardware twin of ``xh430_gripper.py``.

One Dynamixel XH430-V350 (24 V, RS485, Protocol 2.0) drives both blades
through the mirrored linkage, so ONE motor angle is the whole gripper state:
0 deg is the closed rest gap, 10 deg is the working open. The jaw opening that
angle produces comes from ``xh430_gripper.py``'s geometry, so a width in mm means the
same thing on screen and on the bench.

Current-based position control (operating mode 5) rather than plain position
control: closing onto a tube means stalling against it, and a current ceiling
turns that stall into a grip instead of an overload shutdown.

This module is dedicated to THIS gripper -- every parameter lives in the
frozen :class:`Config` below, measured on the bench, not in a config file:

* CLOSING drives the encoder UP and stalls against the mechanical stop at
  ``zero_ticks`` (2088) -- that tick is hinge angle 0, the 4 mm rest gap.
* OPENING drives the encoder DOWN (``sign = -1``) and has **no mechanical
  stop at all**: the linkage runs freely well past its travel. Nothing
  in the hardware protects it, so the ``0..travel_deg`` clamp in
  :meth:`XH430GripperExec.move_to_deg` is the ONLY guard -- do not bypass it
  except for the deliberate grip overshoot.
* The dot on the servo flange marks the CENTRE (2048 = 180 deg), not encoder
  zero; the X series has no physical zero, it turns through a full circle.
  The closed stop sits 40 ticks (3.5 deg) past that centre.

One degree of hinge is only ~2.6 mm of opening, so a small test move is easy
to miss by eye -- use several degrees when checking direction by hand.

Usage::

    py -3.12 coripps/end_effectors/xh430_gripper/exec.py status
    py -3.12 coripps/end_effectors/xh430_gripper/exec.py open
    py -3.12 coripps/end_effectors/xh430_gripper/exec.py close
    py -3.12 coripps/end_effectors/xh430_gripper/exec.py angle 7.5   # hinge degrees
    py -3.12 coripps/end_effectors/xh430_gripper/exec.py width 26    # opening in mm
    py -3.12 coripps/end_effectors/xh430_gripper/exec.py cycle -n 5
    py -3.12 coripps/end_effectors/xh430_gripper/exec.py home        # re-check zero
    py -3.12 coripps/end_effectors/xh430_gripper/exec.py release     # torque off
    py -3.12 coripps/end_effectors/xh430_gripper/exec.py ports       # list adapters

The port is found by the USB adapter's serial number (``usb_serial``), so the
same code works whatever COM number a given PC assigns; ``ports`` shows the
serial number of every adapter.

From other code::

    from coripps.end_effectors.xh430_gripper.exec import XH430GripperExec
    with XH430GripperExec() as gripper:
        gripper.open()
        gripper.close_jaws()
"""

import sys
import time
import argparse
from dataclasses import dataclass

import numpy as np

from one.drivers.dynamixel import (Dynamixel, DynamixelError,
                                   MODE_CURRENT_POSITION, serial_ports)
import coripps.end_effectors.xh430_gripper.xh430_gripper as xh430g


# ---- fixed configuration for THIS gripper ---------------------------------
@dataclass(frozen=True)
class Config:
    """Everything that describes THIS piece of hardware, in one block.

    Frozen and attribute-accessed on purpose: a typo is caught where it is
    written instead of surfacing as a KeyError mid-motion, and the numbers
    cannot be reassigned by accident at runtime. A second gripper is
    ``replace(CONFIG, usb_serial='...')`` rather than an edit here.
    """

    # The FTDI adapter's own serial number -- the same on every PC, unlike
    # the COM number, which is looked up from it at connect time.
    usb_serial: str = 'B003270IA'
    baud: int = 57600
    dxl_id: int = 1

    zero_ticks: int = 2088      # encoder tick where the blades stall shut
    sign: int = -1              # opening DECREASES the encoder count
    # mechanical travel, the software clamp -- FROM the sim model too
    travel_deg: float = float(np.degrees(xh430g.TRAVEL))
    # working open, taken FROM the sim model so the two cannot drift apart
    open_deg: float = float(np.degrees(xh430g.OPEN_TRAVEL))

    # Current ceiling while moving. 2.69 mA per unit, so 200 units is ~540 mA
    # -- enough to close the blades, gentle enough to stall against a tube.
    goal_current: int = 200
    home_current: int = 60      # ~161 mA: gentle creep onto the closed stop
    profile_velocity: int = 60  # 0.229 rev/min per unit; ~14 rpm
    profile_acceleration: int = 20

    # Closing drives this far PAST zero so the blades stall on whatever is
    # there (a tube, or the stop) instead of stopping a hair short of it.
    grip_overshoot_deg: float = 1.0
    move_timeout: float = 3.0

    # Once the close has stalled, the grip is HELD this far past the stall
    # point under this lower ceiling (~270 mA). Holding the close's own goal
    # instead leaves the controller pushing a ~5.5 deg error at the full
    # ceiling indefinitely -- a saturated output the servo latches as an
    # OVERLOAD after a few minutes of holding a tube.
    hold_current: int = 100
    hold_squeeze_deg: float = 0.5


CONFIG = Config()

# Tuning of the settle detector -- about the ALGORITHM, not about the gripper,
# so it stays out of Config.
_SETTLE_TICKS = 2               # movement below this counts as stopped
_SETTLE_POLLS = 4


class XH430GripperExec:
    """The XH430 gripper as open/close verbs instead of registers."""

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.dxl = Dynamixel(usb_serial=cfg.usb_serial, baudrate=cfg.baud,
                             dxl_id=cfg.dxl_id)

    # ---- connection -------------------------------------------------------
    def connect(self, setup=True, check_error=True):
        """``check_error=False`` skips refusing on a latched hardware error
        -- only for ``clear``, which exists to reboot out of one."""
        self.dxl.connect()
        model, firmware = self.dxl.ping()
        print(f'gripper: {self.dxl.model_name(model)} id {self.dxl.dxl_id} '
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
        self.dxl.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def setup(self):
        """Current-based position control, a gentle profile, torque ON.

        Split out of :meth:`connect` so a caller can hold the port open while
        leaving the servo limp -- ``connect(setup=False)`` then ``setup()``
        when it is actually time to move.
        """
        # Operating mode is EEPROM, so THAT one really does need torque off --
        # and only on the first call, since it sticks.
        if self.dxl.read('operating_mode') != MODE_CURRENT_POSITION:
            self.dxl.set_operating_mode(MODE_CURRENT_POSITION)
            print('gripper: switched to current-based position control')
        # The profile registers are RAM: writable with torque already on. Not
        # cycling the torque here is what makes setup() idempotent -- calling
        # it again mid-session must not drop whatever the blades are holding.
        # (profile_velocity 0 means FULL SPEED, which is how a factory-fresh
        # servo ships, so it is always written before the first move.)
        self.dxl.write('profile_velocity', self.cfg.profile_velocity)
        self.dxl.write('profile_acceleration', self.cfg.profile_acceleration)
        if not self.dxl.read('torque_enable'):
            self.dxl.torque(True)
        self.dxl.write('goal_current', self.cfg.goal_current)

    def release(self):
        """Torque off -- the blades go limp."""
        self.dxl.torque(False)
        print('gripper: torque off')

    # ---- state ------------------------------------------------------------
    def angle_deg(self):
        """Present hinge angle in degrees (0 = closed against the stop)."""
        return self.dxl.ticks_to_deg(
            self.dxl.read('present_position') - self.cfg.zero_ticks
        ) * self.cfg.sign

    def opening_mm(self):
        q = np.deg2rad(self.angle_deg())
        return xh430g.XH430Gripper.opening_from_angle(q) * 1000.0

    def status(self):
        present = self.dxl.read('present_position')
        line = (f'angle {self.angle_deg():6.2f} deg  '
                f'opening {self.opening_mm():5.2f} mm  '
                f'encoder {present}  '
                f'torque {"on" if self.dxl.read("torque_enable") else "off"}  '
                f'current {self.dxl.present_current_ma():6.1f} mA  '
                f'{self.dxl.read("present_input_voltage") * 0.1:.1f} V  '
                f'{self.dxl.read("present_temperature")} C')
        print(line)
        return line

    # ---- motion -----------------------------------------------------------
    def move_to_deg(self, deg, wait=True, clamp=True):
        """Drive the hinge to ``deg`` -- 0 is closed, ``OPEN_DEG`` is open.

        ``clamp`` keeps the target inside 0..TRAVEL_DEG. It is the only thing
        standing between a wrong number and an over-opened linkage, since the
        opening direction has no mechanical stop; the grip is the one caller
        that switches it off, to push past zero.
        """
        travel = self.cfg.travel_deg
        target = float(np.clip(deg, 0.0, travel)) if clamp else float(deg)
        if target != deg:
            print(f'gripper: {deg:.2f} deg clamped to {target:.2f} '
                  f'(travel 0..{travel:g})')
        ticks = self.cfg.zero_ticks + \
            self.dxl.deg_to_ticks(target) * self.cfg.sign
        self.dxl.write('goal_current', self.cfg.goal_current)
        self.dxl.write('goal_position', int(ticks))
        if wait:
            self._settle()
        return target

    def _settle(self, timeout=None):
        """Block until the position goes quiet; return ``(ticks, mA)``.

        Position, not the ``moving`` flag: a grip never reaches its goal, so
        ``moving`` would stay set until the timeout on every single close.
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

    def move_to_mm(self, width_mm, wait=True):
        """Out-of-range widths end up clamped by :meth:`move_to_deg`."""
        q = xh430g.XH430Gripper.angle_from_opening(width_mm / 1000.0)
        return self.move_to_deg(np.degrees(q), wait=wait)

    def open(self, wait=True):
        """Open to the WORKING angle, not the mechanical limit."""
        return self.move_to_deg(self.cfg.open_deg, wait=wait)

    def close_jaws(self, wait=True):
        """Close all the way and hold.

        Closing IS a stall -- against the tube if there is one, against the
        mechanical stop if there is not -- so the goal is driven a degree past
        zero and the current ceiling turns that stall into a grip. Once it has
        settled the grip is eased into :meth:`hold`. With ``wait=False`` the
        caller has to call :meth:`hold` itself once the blades have stopped.
        """
        target = self.move_to_deg(-self.cfg.grip_overshoot_deg, wait=wait,
                                  clamp=False)
        if wait:
            self.hold()
        return target

    def hold(self):
        """Hold the settled grip with a small squeeze and a lower ceiling.

        Re-aims the goal ``hold_squeeze_deg`` past where the blades actually
        stopped, under ``hold_current``: the controller keeps a small error
        instead of a saturated one, so a long hold does not trip the servo's
        overload protection. Returns the new goal angle.
        """
        target = self.angle_deg() - self.cfg.hold_squeeze_deg
        ticks = self.cfg.zero_ticks + \
            self.dxl.deg_to_ticks(target) * self.cfg.sign
        self.dxl.write('goal_current', self.cfg.hold_current)
        self.dxl.write('goal_position', int(ticks))
        return target

    def find_closed_stop(self):
        """Creep shut at a low current until the blades stall on the stop.

        That stall point IS hinge angle 0, so this is how ``ZERO_TICKS`` was
        measured; re-run it whenever the horn is re-seated.
        """
        self.dxl.write('goal_current', self.cfg.home_current)
        start = self.dxl.read('present_position')
        # a few degrees past the far end of the travel: it stalls long before
        beyond = self.dxl.deg_to_ticks(self.cfg.travel_deg + 5.0)
        self.dxl.write('goal_position', int(start - beyond * self.cfg.sign))
        ticks, current = self._settle(timeout=6.0)
        self.dxl.write('goal_current', self.cfg.goal_current)
        return ticks, current


# ---------------------------------------------------------------------------
def _cmd_status(gripper, args):
    gripper.status()


def _cmd_open(gripper, args):
    gripper.open()
    gripper.status()


def _cmd_close(gripper, args):
    gripper.close_jaws()
    gripper.status()


def _cmd_angle(gripper, args):
    gripper.move_to_deg(args.degrees)
    gripper.status()


def _cmd_width(gripper, args):
    gripper.move_to_mm(args.millimeters)
    gripper.status()


def _cmd_cycle(gripper, args):
    for i in range(args.n):
        gripper.open()
        opened = (gripper.angle_deg(), gripper.opening_mm(),
                  gripper.dxl.present_current_ma())
        time.sleep(args.dwell)
        gripper.close_jaws()
        shut = (gripper.angle_deg(), gripper.opening_mm(),
                gripper.dxl.present_current_ma())
        time.sleep(args.dwell)
        print(f'cycle {i + 1}/{args.n}  '
              f'open {opened[0]:5.2f} deg / {opened[1]:5.2f} mm '
              f'@ {opened[2]:4.0f} mA   ->   '
              f'shut {shut[0]:5.2f} deg / {shut[1]:5.2f} mm '
              f'@ {shut[2]:4.0f} mA')
    gripper.status()


def _cmd_home(gripper, args):
    ticks, current = gripper.find_closed_stop()
    zero = gripper.cfg.zero_ticks
    print(f'gripper: stalled shut at encoder {ticks} at {current:.0f} mA '
          f'(zero_ticks is {zero}, off by {ticks - zero:+d})')
    if abs(ticks - zero) > 4:
        print(f'         the mechanism has shifted -- set zero_ticks = {ticks} '
              f'in Config at the top of this file')
    gripper.status()


def _cmd_release(gripper, args):
    gripper.release()


def _cmd_clear(gripper, args):
    bits, desc = gripper.dxl.hardware_error()
    print(f'hardware error 0x{bits:02x}: {desc}')
    gripper.dxl.clear_hardware_error()
    print('gripper: rebooted')


def _cmd_ports(args):
    for device, sn, desc in serial_ports():
        mark = '  <- gripper' if sn == CONFIG.usb_serial else ''
        print(f'{device:6s}  {sn or "-":12s}  {desc}{mark}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)

    def add(name, help_text, **defaults):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(**defaults)
        return p

    add('status', 'read angle / opening / current', func=_cmd_status,
        setup=False)
    add('open', f'open to {CONFIG.open_deg:g} deg', func=_cmd_open)
    add('close', 'close and grip', func=_cmd_close)

    p = add('angle', 'go to a hinge angle in degrees', func=_cmd_angle)
    p.add_argument('degrees', type=float)

    p = add('width', 'go to a jaw opening in mm', func=_cmd_width)
    p.add_argument('millimeters', type=float)

    p = add('cycle', 'open/close repeatedly', func=_cmd_cycle)
    p.add_argument('-n', type=int, default=3)
    p.add_argument('--dwell', type=float, default=0.5)

    add('home', 'creep onto the closed stop and check ZERO_TICKS',
        func=_cmd_home)
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
    gripper = XH430GripperExec()
    try:
        gripper.connect(setup=getattr(args, 'setup', True),
                       check_error=getattr(args, 'check_error', True))
        args.func(gripper, args)
    except DynamixelError as exc:
        print(f'gripper: {exc}')
        return 1
    finally:
        gripper.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
