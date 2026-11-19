"""Reset the robots after a run stopped half way: both arms home.

The sampler home (``home_sampler``), then the picker (``home_picker``) --
each planned around the other where it stands. One that cannot go (blocked
by the other, a path not found) is tried again once after the other has
moved. The end effectors are left as they are (``reset_ee.py``).

Headless, nothing to press.

    py -3.12 coripps/tool/reset_robot.py
"""

import sys

from coripps.tool import home_picker, home_sampler

ARMS = (('sampler', home_sampler.home), ('picker', home_picker.home))
ROUNDS = 2


def reset():
    """Both arms home; True when both arrived."""
    left = list(ARMS)
    for k in range(ROUNDS):
        failed = []
        for name, home in left:
            print(f'== {name} home' + (' (again)' if k else ''), flush=True)
            try:
                ok = home()
            except Exception as exc:        # one arm's fault: the other goes
                print(f'  {type(exc).__name__}: {exc}')
                ok = False
            if not ok:
                failed.append((name, home))
        # nothing moved this round: another would plan the same
        if not failed or len(failed) == len(left):
            left = failed
            break
        left = failed
    print('reset: both arms home' if not left else
          f"reset: FAILED -- {', '.join(n for n, _ in left)} not home")
    return not left


if __name__ == '__main__':
    sys.exit(0 if reset() else 1)
