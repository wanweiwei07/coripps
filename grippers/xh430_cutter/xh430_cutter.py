"""XH430-driven four-bar cutter.

One servo turns link1; link2 and link3 close a planar four-bar with the base,
so they follow link1 through NONLINEAR mimics (``mmc=(src, fn)``).

Four-bar, in the base X-Z plane (all three axes are local +Y, mm); link1 is
the driver, q counter-clockwise about +Y, travel 0..160 deg:
    A  base  -> link1 pivot   (14.25, 51.25)
    B  link1 -> link2 pivot   (5.534, 35.501) at q = 0
    D  base  -> link3 pivot   (14.25, 27.25)
    C  link2 <-> link3 pin     (closes the loop)
    AB = 18 (link1), BC = 24 (link2), DC = 24 (link3), AD = 24 (base)

At q = 0 link1 and link2 are collinear, so C0 = A + (B0 - A) * (AB + BC) / AB.
Grashof: 18 + 24 <= 24 + 24, shortest link next to the ground -> crank-rocker,
link1 turns all the way round and link3 rocks. q = 0 is link3's extreme.

Mimic, for link1's angle q1 about +Y (Ry(q) adds q to phi(v) = atan2(v_x, v_z)):
    B(q1) = A + Ry(q1) (B0 - A)
    C(q1) = intersection of circle(B, BC) and circle(D, DC), on C0's side of BD:
            d = |D - B|,  a = (BC^2 - DC^2 + d^2) / (2 d),  h = sqrt(BC^2 - a^2)
            C = B + a u + s h n,   u = (D - B) / d,  n = u rotated +90 deg
    q3 = phi(C - D) - phi(C0 - D)              link3 vs base
    q2 = phi(C - B) - phi(C0 - B0) - q1        link2 vs link1
BD stays in [AD - AB, AD + AB] = [6, 42] < BC + DC, so the circles always meet
and the branch side s never flips.
"""

import os

import numpy as np

import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.base.mech_base as orbmb


# pivots in the base X-Z plane (m); each link mesh is authored about its pivot
_A = np.array([0.01425, 0.05125])
_B0 = np.array([0.005534, 0.035501])
_D = np.array([0.01425, 0.02725])
_Y1, _Y2, _Y3 = 0.002, 0.006, 0.006    # pivot y: link1 on base, link2 on link1,
                                       # link3 on base
_L1 = float(np.linalg.norm(_B0 - _A))  # 18 mm
_L2 = 0.024
_C0 = _A + (_B0 - _A) * (_L1 + _L2) / _L1  # link1, link2 collinear at q = 0
_L3 = float(np.linalg.norm(_C0 - _D))      # 24 mm
_TRAVEL = np.deg2rad(157.0)                # link1, ccw about +Y


def _phi(v):
    """Angle of an X-Z vector about +Y."""
    return np.arctan2(v[0], v[1])


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _ry(v, q):
    c, s = np.cos(q), np.sin(q)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


def _pin(b, s):
    """Loop-closing pin C: circle(b, L2) and circle(D, L3), on side ``s``."""
    dv = _D - b
    d = np.linalg.norm(dv)
    u = dv / d
    a = (_L2 ** 2 - _L3 ** 2 + d ** 2) / (2.0 * d)
    h = np.sqrt(max(_L2 ** 2 - a ** 2, 0.0))
    return b + a * u + s * h * np.array([-u[1], u[0]])


# which side of BD the pin sits on, fixed by the assembled q = 0 pose
_SIDE = 1.0 if np.linalg.norm(_pin(_B0, 1.0) - _C0) < np.linalg.norm(
    _pin(_B0, -1.0) - _C0) else -1.0


def _linkage(q1):
    b = _A + _ry(_B0 - _A, q1)
    return b, _pin(b, _SIDE)


def mmc_link2(q1):
    """link2 angle relative to link1, driven by link1's ``q1``."""
    b, c = _linkage(q1)
    return float(_wrap(_phi(c - b) - _phi(_C0 - _B0) - q1))


def mmc_link3(q1):
    """link3 angle relative to the base, driven by link1's ``q1``."""
    _, c = _linkage(q1)
    return float(_wrap(_phi(c - _D) - _phi(_C0 - _D)))


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'base.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER,
    )
    link1 = orbms.Link.from_file(
        os.path.join(mesh_dir, 'link1.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE,
    )
    link2 = orbms.Link.from_file(
        os.path.join(mesh_dir, 'link2.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SALMON_PINK,
    )
    link3 = orbms.Link.from_file(
        os.path.join(mesh_dir, 'link3.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE,
    )

    jnt1 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=link1,
        axis=ouc.StandardAxis.Y,
        pos=np.array([_A[0], _Y1, _A[1]], dtype=np.float32),
        lmt_lo=0.0,
        lmt_up=_TRAVEL,
    )
    jnt2 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=link1,
        child_lnk=link2,
        axis=ouc.StandardAxis.Y,
        pos=np.array([_B0[0] - _A[0], _Y2, _B0[1] - _A[1]], dtype=np.float32),
        mmc=(jnt1, mmc_link2),
    )
    jnt3 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=link3,
        axis=ouc.StandardAxis.Y,
        pos=np.array([_D[0], _Y3, _D[1]], dtype=np.float32),
        mmc=(jnt1, mmc_link3),
    )

    structure.add_lnk(base_lnk)
    structure.add_lnk(link1)
    structure.add_lnk(link2)
    structure.add_lnk(link3)
    structure.add_jnt(jnt1)
    structure.add_jnt(jnt2)
    structure.add_jnt(jnt3)
    structure.ignore_collision(link2, link3)  # pinned together at C
    structure.compile()
    return structure


class XH430Cutter(orbmb.MechBase):
    """Four-bar cutter: one active joint (link1); link2 / link3 are mimics."""

    @classmethod
    def _build_structure(cls):
        return prepare_ms()


if __name__ == '__main__':
    import builtins

    import one.viewer.world as ovw

    base = ovw.World(cam_pos=[0.0, 0.25, 0.04], cam_lookat_pos=[0, 0, 0.04])
    cutter = XH430Cutter()
    cutter.fk(qs=[0.0])
    cutter.attach_to(base.scene)

    speed = np.deg2rad(60.0)   # rad/s, sweeps 0 -> 160 -> 0 deg
    state = {'q': 0.0, 'dir': 1.0}

    def tick(dt):
        # if state['q'] == _TRAVEL:
        #     state['dir'] = 0
        q = state['q'] + state['dir'] * speed * dt
        if q >= _TRAVEL or q <= 0.0:
            state['dir'] *= -1.0
            q = float(np.clip(q, 0.0, _TRAVEL))
        state['q'] = q
        cutter.fk(qs=[q])

    base.schedule_interval(tick, interval=1 / 30.0)
    builtins.base = base
    builtins.cutter = cutter
    base.run()
