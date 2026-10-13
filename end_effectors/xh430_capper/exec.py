"""Drive the REAL XH430 tube capper -- opens and presses shut a tube cap.

One Dynamixel XH430 (RS485, Protocol 2.0), fixed in the cell. There is no sim
model: the capper is only ever driven, never planned around, so everything is
in the servo's own dial degrees (0..360 over one turn):

* HOME is 180 deg -- parked, waiting for the picker to place a tube.
* STOPPER is 150 deg.
* OPEN is 275 deg.
* CLOSED is 75 deg, and closing is a PRESS: the goal is driven
  ``press_overshoot_deg`` past it, so the arm stalls on the cap and the
  current ceiling turns that stall into a held press -- the same trick as the
  gripper's grip. Without a cap it simply stops at the overshoot target.

Current-based position control (operating mode 5), so a stall is a press
instead of an overload shutdown.

Usage::

    py -3.12 coripps/end_effectors/xh430_capper/exec.py status
    py -3.12 coripps/end_effectors/xh430_capper/exec.py home
    py -3.12 coripps/end_effectors/xh430_capper/exec.py stopper
    py -3.12 coripps/end_effectors/xh430_capper/exec.py open
    py -3.12 coripps/end_effectors/xh430_capper/exec.py close
    py -3.12 coripps/end_effectors/xh430_capper/exec.py release   # torque off
    py -3.12 coripps/end_effectors/xh430_capper/exec.py ports     # adapters

The port is found by the USB adapter's serial number (``usb_serial``), so the
same code works whatever COM number a given PC assigns.

From other code::

    from coripps.end_effectors.xh430_capper.exec import XH430CapperExec
    with XH430CapperExec() as capper:
        capper.set_opening('home')    # the picker places the tube
        capper.set_opening('open')
        capper.set_opening('close')   # press the cap shut
"""

import sys
import time
import argparse
from dataclasses import dataclass

from one.drivers.dynamixel import (Dynamixel, DynamixelError,
                                   MODE_CURRENT_POSITION, serial_ports)


# ---- fixed configuration for THIS capper ----------------------------------
@dataclass(frozen=True)
class Config:
    """Everything that describes THIS piece of hardware, in one block."""

    # The FTDI adapter's own serial number -- the same on every PC, unlike
    # the COM number, which is looked up from it at connect time.
    usb_serial: str = 'B0031TWGA'
    baud: int = 57600
    dxl_id: int = 1

    # Servo-dial angles (deg).
    home_deg: float = 180.0     # parked, waiting for the picker's tube
    stopper_deg: float = 150.0
    open_deg: float = 275.0
    close_deg: float = 75.0
    # Closing drives this far PAST close_deg so it presses on the cap instead
    # of stopping a hair short of it. Not yet tuned on the bench.
    press_overshoot_deg: float = 3.0

    # Current ceiling, which is also the press force. 2.69 mA per unit, so
    # 200 units is ~540 mA. Not yet tuned on the bench.
    goal_current: int = 200
    profile_velocity: int = 60  # 0.229 rev/min per unit; ~14 rpm
    profile_acceleration: int = 20
    move_timeout: float = 7.0


CONFIG = Config()

# Tuning of the settle detector -- about the ALGORITHM, not about the capper,
# so it stays out of Config.
_SETTLE_TICKS = 2               # movement below this counts as stopped
_SETTLE_POLLS = 4


class XH430CapperExec:
    """The XH430 capper as open/close verbs instead of registers."""

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.dxl = Dynamixel(usb_serial=cfg.usb_serial, baudrate=cfg.baud,
                             dxl_id=cfg.dxl_id)

    # ---- connection -------------------------------------------------------
    def connect(self, setup=True):
        self.dxl.connect()
        model, firmware = self.dxl.ping()
        print(f'capper: {self.dxl.model_name(model)} id {self.dxl.dxl_id} '
              f'on {self.dxl.port} (firmware {firmware})')
        bits, desc = self.dxl.hardware_error()
        if bits:
            raise DynamixelError(
                f'servo reports a latched hardware error: {desc} -- power '
                f'cycle it')
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

        Idempotent: torque is only cycled when the (EEPROM) operating mode
        actually has to change, so calling it again mid-session does not drop
        a press.
        """
        if self.dxl.read('operating_mode') != MODE_CURRENT_POSITION:
            self.dxl.set_operating_mode(MODE_CURRENT_POSITION)
            print('capper: switched to current-based position control')
        # profile_velocity 0 means FULL SPEED (factory default), so it is
        # always written before the first move.
        self.dxl.write('profile_velocity', self.cfg.profile_velocity)
        self.dxl.write('profile_acceleration', self.cfg.profile_acceleration)
        if not self.dxl.read('torque_enable'):
            self.dxl.torque(True)
        self.dxl.write('goal_current', self.cfg.goal_current)

    def release(self):
        """Torque off -- the arm goes limp."""
        self.dxl.torque(False)
        print('capper: torque off')

    # ---- state ------------------------------------------------------------
    def servo_deg(self):
        """Present angle on the servo's own dial."""
        return self.dxl.ticks_to_deg(self.dxl.read('present_position'))

    def status(self):
        line = (f'servo {self.servo_deg():7.2f} deg  '
                f'encoder {self.dxl.read("present_position")}  '
                f'torque {"on" if self.dxl.read("torque_enable") else "off"}  '
                f'current {self.dxl.present_current_ma():6.1f} mA  '
                f'{self.dxl.read("present_input_voltage") * 0.1:.1f} V  '
                f'{self.dxl.read("present_temperature")} C')
        print(line)
        return line

    # ---- motion -----------------------------------------------------------
    def _move_to_servo_deg(self, deg, wait=True):
        self.dxl.write('goal_current', self.cfg.goal_current)
        self.dxl.write('goal_position', self.dxl.deg_to_ticks(deg))
        if wait:
            self._settle()
        return deg

    def _settle(self):
        """Block until the position goes quiet; return ``(ticks, mA)``.

        Position, not the ``moving`` flag: a press never reaches its goal, so
        ``moving`` would stay set until the timeout on every close.
        """
        last = self.dxl.read('present_position')
        quiet = 0
        deadline = time.time() + self.cfg.move_timeout
        while time.time() < deadline:
            time.sleep(0.05)
            pos = self.dxl.read('present_position')
            quiet = quiet + 1 if abs(pos - last) <= _SETTLE_TICKS else 0
            last = pos
            if quiet >= _SETTLE_POLLS:
                break
        return last, self.dxl.present_current_ma()

    def set_opening(self, mode, wait=True):
        """Move to ``'home'``, ``'stopper'``, ``'open'`` or ``'close'``.

        ``'home'`` parks where the picker places the tube; ``'close'`` presses
        the cap shut and holds: it drives ``press_overshoot_deg`` past
        ``close_deg``, so the arm stalls on the cap and the current ceiling
        holds the press. (Not to be confused with :meth:`close`, which only
        closes the serial port.)
        """
        if mode == 'home':
            target = self.cfg.home_deg
        elif mode == 'stopper':
            target = self.cfg.stopper_deg
        elif mode == 'open':
            target = self.cfg.open_deg
        elif mode == 'close':
            direction = 1.0 if self.cfg.close_deg > self.cfg.open_deg else -1.0
            target = (self.cfg.close_deg +
                      direction * self.cfg.press_overshoot_deg)
        else:
            raise ValueError(f"mode must be 'home', 'stopper', 'open' or "
                             f"'close', got {mode!r}")
        return self._move_to_servo_deg(target, wait=wait)


# ---------------------------------------------------------------------------
def _cmd_status(capper, args):
    capper.status()


def _cmd_move(capper, args):
    capper.set_opening(args.mode)
    capper.status()


def _cmd_release(capper, args):
    capper.release()


def _cmd_ports(args):
    for device, sn, desc in serial_ports():
        mark = '  <- capper' if sn == CONFIG.usb_serial else ''
        print(f'{device:6s}  {sn or "-":12s}  {desc}{mark}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)

    def add(name, help_text, **defaults):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(**defaults)
        return p

    add('status', 'read angle / current', func=_cmd_status, setup=False)
    add('home', f'park at {CONFIG.home_deg:g} deg for the picker',
        func=_cmd_move, mode='home')
    add('stopper', f'go to the stopper position ({CONFIG.stopper_deg:g} deg)',
        func=_cmd_move, mode='stopper')
    add('open', f'swing open to {CONFIG.open_deg:g} deg',
        func=_cmd_move, mode='open')
    add('close', f'press the cap shut ({CONFIG.close_deg:g} deg)',
        func=_cmd_move, mode='close')
    add('release', 'torque off', func=_cmd_release, setup=False)
    add('ports', 'list serial ports and their USB serial numbers',
        func=_cmd_ports, offline=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, 'offline', False):         # needs no servo
        args.func(args)
        return 0
    capper = XH430CapperExec()
    try:
        capper.connect(setup=getattr(args, 'setup', True))
        args.func(capper, args)
    except DynamixelError as exc:
        print(f'capper: {exc}')
        return 1
    finally:
        capper.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
