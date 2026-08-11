import os
import sys
import numpy as np

# Allow running as `python coripps/real_state.py` from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.drivers.cobotta import BCapError, Cobotta

from coripps.robots.mobisys import mobisys


# This PC is 192.168.0.100; the two COBOTTA controllers are .101 and .102.
_TUBE_HANDLER_HOST = '192.168.0.102'
_LEAF_SAMPLER_HOST = '192.168.0.101'
_POLL_INTERVAL = 0.1
_PRINT_THRESHOLD = np.deg2rad(0.1)


def connect_cobotta(host, label):
    """Connect to one controller; return None (and say so) when unreachable."""
    try:
        cobotta = Cobotta(host).connect()
    except (OSError, BCapError) as exc:
        print(f'{label}: cannot read {host} ({exc}) -- left at its sim pose')
        return None
    print(f'{label}: reading real state from {host}')
    return cobotta


def show_real_state():
    base = ovw.World(cam_pos=(1.25, 0.75, 0.65),
                     cam_lookat_pos=(0.45, -0.10, 0.12))
    ossop.frame().attach_to(base.scene)

    mobisys_base, tube_handler, leaf_sampler = mobisys()
    mobisys_base.attach_to(base.scene)
    tube_handler.attach_to(base.scene)
    tube_handler.gripper.attach_to(base.scene)
    tube_handler.gripper.toggle_tcp('grasp_center', color_mat=ouc.CoordColor.MYC)
    leaf_sampler.attach_to(base.scene)
    leaf_sampler.toggle_tcp(
        'd405',
        length_scale=0.3,
        radius_scale=0.5,
        color_mat=ouc.CoordColor.MYC,
    )

    links = [
        {
            'label': 'tube_handler',
            'robot': tube_handler,
            'cobotta': connect_cobotta(_TUBE_HANDLER_HOST, 'tube_handler'),
            'last_qs': None,
        },
        {
            'label': 'leaf_sampler',
            'robot': leaf_sampler,
            'cobotta': connect_cobotta(_LEAF_SAMPLER_HOST, 'leaf_sampler'),
            'last_qs': None,
        },
    ]

    def _poll(link):
        """Read one controller and drive its sim robot. False = drop the link."""
        cobotta = link['cobotta']
        robot = link['robot']
        try:
            qs = cobotta.joint_angles(ndof=robot.ndof)
        except (OSError, BCapError) as exc:
            print(f"{link['label']}: lost {cobotta.host} ({exc}) -- polling stopped")
            cobotta.close()
            return False
        robot.fk(qs=qs)
        last_qs = link['last_qs']
        if last_qs is None or np.any(np.abs(qs - last_qs) > _PRINT_THRESHOLD):
            link['last_qs'] = qs
            print(f"{link['label']}: qs_deg={np.round(np.degrees(qs), 2)}")
        return True

    def _update(dt):
        for link in links:
            if link['cobotta'] is None:
                continue
            if not _poll(link):
                link['cobotta'] = None

    base.schedule_interval(_update, interval=_POLL_INTERVAL)
    return base, mobisys_base, tube_handler, leaf_sampler, links


def close_links(links):
    for link in links:
        if link['cobotta'] is not None:
            link['cobotta'].close()
            link['cobotta'] = None


if __name__ == '__main__':
    import builtins

    base, mobisys_base, tube_handler, leaf_sampler, links = show_real_state()
    builtins.base = base
    builtins.mobisys_base = mobisys_base
    builtins.tube_handler = tube_handler
    builtins.leaf_sampler = leaf_sampler
    builtins.links = links
    try:
        base.run()
    finally:
        close_links(links)
