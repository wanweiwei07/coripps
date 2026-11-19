"""Send the capper to its home dial angle: where the picker places the tube.

Headless: no viewer, nothing to press. The capper's own command line does
the move (``end_effectors/xh430_capper/exec.py home``); this is that, next to
``home_picker`` / ``home_sampler``.

    py -3.12 coripps/tool/home_capper.py
"""

import sys

from coripps.end_effectors.xh430_capper import exec as capper_exec


if __name__ == '__main__':
    sys.exit(capper_exec.main(['home']))
