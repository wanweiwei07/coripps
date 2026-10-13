"""CONTEC DIO-1616LN-USB digital output driver (vacuum / blow / ultrasonic cutter).

Talks to CONTEC's API-USBP(WDM) runtime through ``cdio.dll`` with ctypes, so
the CONTEC driver has to be installed and the board registered (default device
name ``DIO000``, see the Device Manager / API-TOOL configuration).

Output port 0 wiring::

    bit 0  vacuum           ON = suction,  OFF = suction stopped
    bit 1  vacuum break     ON = blow,     OFF = blow stopped
    bit 2  relay            ON = ultrasonic cutter running, OFF = stopped

Bits 0 and 1 drive the same vacuum generator, so they are interlocked: turning
one on always turns the other off first, and both are never ON together.

Outputs are sink-type open collector with CONTEC negative logic: writing 1
switches the transistor on and pulls OUTxx to N-COM.
"""

import ctypes

_PORT = 0
_BIT_VACUUM = _PORT * 8 + 0
_BIT_BLOW = _PORT * 8 + 1
_BIT_CUTTER = _PORT * 8 + 2

DIO_ERR_SUCCESS = 0


class ContecDIOError(RuntimeError):
    pass


class ContecDIO:

    def __init__(self, device_name='DIO000', dll_name='cdio.dll'):
        self._dll = ctypes.WinDLL(dll_name)
        self._dll.DioInit.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_short)]
        self._dll.DioExit.argtypes = [ctypes.c_short]
        self._dll.DioOutBit.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.c_ubyte]
        self._dll.DioEchoBackBit.argtypes = [ctypes.c_short, ctypes.c_short,
                                             ctypes.POINTER(ctypes.c_ubyte)]
        self._dll.DioGetErrorString.argtypes = [ctypes.c_long, ctypes.c_char_p]
        for fn in (self._dll.DioInit, self._dll.DioExit, self._dll.DioOutBit,
                   self._dll.DioEchoBackBit, self._dll.DioGetErrorString):
            fn.restype = ctypes.c_long
        self._id = None
        dev_id = ctypes.c_short()
        self._check(self._dll.DioInit(device_name.encode('ascii'), ctypes.byref(dev_id)),
                    'DioInit(%s)' % device_name)
        self._id = dev_id.value

    def _check(self, ret, what):
        if ret != DIO_ERR_SUCCESS:
            buf = ctypes.create_string_buffer(256)
            self._dll.DioGetErrorString(ret, buf)
            raise ContecDIOError('%s failed (%d): %s'
                                 % (what, ret, buf.value.decode('mbcs', 'replace')))

    def _out(self, bit, on):
        self._check(self._dll.DioOutBit(self._id, bit, 1 if on else 0),
                    'DioOutBit(%d)' % bit)

    def _echo(self, bit):
        data = ctypes.c_ubyte()
        self._check(self._dll.DioEchoBackBit(self._id, bit, ctypes.byref(data)),
                    'DioEchoBackBit(%d)' % bit)
        return bool(data.value)

    def vacuum_generator(self, mode):
        """Set the vacuum generator to 'suck', 'blow' or 'off'.

        Bits 0/1 are interlocked: the bit being released is always cleared
        before the other one is set, so both are never ON together.
        """
        if mode not in ('suck', 'blow', 'off'):
            raise ValueError("mode must be 'suck', 'blow' or 'off', got %r" % (mode,))
        if mode != 'blow':
            self._out(_BIT_BLOW, False)
        if mode != 'suck':
            self._out(_BIT_VACUUM, False)
        if mode == 'suck':
            self._out(_BIT_VACUUM, True)
        elif mode == 'blow':
            self._out(_BIT_BLOW, True)

    def ultrasonic_cutter(self, on):
        """on=True closes the relay and starts the ultrasonic cutter."""
        self._out(_BIT_CUTTER, on)

    def state(self):
        """Current output states read back from the board."""
        suck, blow = self._echo(_BIT_VACUUM), self._echo(_BIT_BLOW)
        return {'vacuum_generator': 'suck' if suck else 'blow' if blow else 'off',
                'ultrasonic_cutter': self._echo(_BIT_CUTTER)}

    def close(self):
        """Switch every output off and release the device."""
        if self._id is None:
            return
        try:
            self.ultrasonic_cutter(False)
            self.vacuum_generator('off')
        finally:
            self._dll.DioExit(self._id)
            self._id = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
