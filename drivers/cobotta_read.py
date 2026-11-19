"""Read-only access to the COBOTTA controllers: a connection, or a one-off
read of the joints, None (and said why) when the controller is unreachable
-- the caller leaves that robot at its sim pose. Nothing is sent to them."""

from one.drivers.cobotta import BCapError, Cobotta


def connect(host, label):
    """A connection to ``host`` (``label``: its name in the messages), or
    None when it cannot be reached."""
    try:
        cobotta = Cobotta(host).connect()
    except (OSError, BCapError) as exc:
        print(f'{label}: cannot read {host} ({exc}) -- left at its sim pose')
        return None
    print(f'{label}: reading {host}')
    return cobotta


def read_qs(host, ndof=6, label=None):
    """``host``'s joint angles (rad) by a connection of its own, or None
    when it cannot be reached."""
    try:
        with Cobotta(host) as cobotta:
            return cobotta.joint_angles(ndof=ndof)
    except (OSError, BCapError) as exc:
        print(f'{label or host}: cannot read {host} ({exc}) -- left at its '
              'sim pose')
        return None
