"""Solve ``layout_cap`` into ``config/poses_cap.json``.

Headless, like ``precompute_poses_pick``: this PRODUCES the artifact. Every
pose gets its IK and every straight-line path its samples (a branch flip or a
collision anywhere fails it), against the cell model with the sampler at its
sim pose; ``pick_cap_store`` re-checks the paths against the real sampler
pose before it runs them.

Re-run it after editing ``layout_cap`` (the executor refuses a file solved
from different values).

    py -3.12 coripps/tool/tube/precompute_poses_cap.py
"""

import sys
import argparse

import numpy as np

import coripps.picker_motion as pkm
import coripps.tool.tube.layout_pick as layout
import coripps.tool.pose_sequence as pseq
import coripps.tool.tube.layout_cap as cap
import coripps.tool.tube.poses_io_cap as cpio
from coripps.robots.mobisys import mobisys


def solve_pose(picker, pln_ctx, pos, rotmat, near, what):
    """Joint config putting ``grasp_center`` at ``pos`` with ``rotmat``,
    nearest ``near``; None (with the reason printed) on no IK / collision."""
    qs_list = picker.ik(tgt_pos=pos, tgt_rotmat=rotmat,
                        tcp=picker.gripper.tcp('grasp_center'),
                        max_solutions=8)
    valid = [np.asarray(q, dtype=np.float32) for q in (qs_list or [])
             if pln_ctx.is_state_valid(pln_ctx.enforce_bounds(q))]
    if not valid:
        print(f'  {what} {np.round(pos * 1000, 2)} mm: no IK, or every '
              'solution is in collision or out of the joint limits')
        return None
    return min(valid, key=lambda q: float(np.linalg.norm(q - near)))


def solve_line(picker, pln_ctx, start_qs, start_pos, rotmat, delta, what):
    """A straight TCP line from ``start_pos`` by the vector ``delta``,
    holding ``rotmat``, on ``start_qs``'s IK branch, ``start_qs`` first;
    None (with the reason printed) on no IK, a branch flip or a collision."""
    delta = np.asarray(delta, dtype=np.float32)
    n = int(np.ceil(float(np.linalg.norm(delta)) / layout.PATH_STEP - 1e-3))
    positions = [start_pos + delta * k / n for k in range(1, n + 1)]
    path = pseq.follow_positions(picker, start_qs, positions, rotmat)
    where = (f'{np.round(delta * 1000, 1)} mm from '
             f'{np.round(start_pos * 1000, 1)} mm')
    if path is None:
        print(f'  {what} {where}: no smooth IK all the way')
        return None
    if not pkm.local_sequence_valid(pln_ctx, path):
        print(f'  {what} {where}: in collision')
        return None
    return path


def solve_descent(picker, pln_ctx, start_qs, start_pos, rotmat, dist, what):
    """:func:`solve_line` straight down by ``dist`` (up when negative)."""
    return solve_line(picker, pln_ctx, start_qs, start_pos, rotmat,
                      np.array([0.0, 0.0, -dist], dtype=np.float32), what)


def least_j6(picker, pln_ctx, pos, rotmat):
    """Of the valid IK solutions at ``(pos, rotmat)``, the one with the
    least J6 -- what a drop IK is chosen near. None when there is none."""
    sols = [q for q in (picker.ik(tgt_pos=pos, tgt_rotmat=rotmat,
                                  tcp=picker.gripper.tcp('grasp_center'),
                                  max_solutions=8) or [])
            if pln_ctx.is_state_valid(pln_ctx.enforce_bounds(q))]
    return min(sols, key=lambda q: abs(float(q[5]))) if sols else None


def precompute(path=cpio.CAP_POSES_JSON):
    """Solve everything and save it. False when anything fails."""
    _, picker, sampler = mobisys()
    pln_ctx, _ = pkm.build_planning_context(picker, sampler)

    orientations = {}
    for name, (rotmat, drop_pos) in cap.PLACINGS.items():
        print(f'{name}: drop at {np.round(drop_pos * 1000, 2)} mm, TCP x '
              f'{np.round(np.asarray(rotmat)[:, 0], 3)}')
        near = least_j6(picker, pln_ctx, drop_pos, rotmat)
        drop = None if near is None else solve_pose(
            picker, pln_ctx, drop_pos, rotmat, near, 'drop point')
        place = None if drop is None else solve_descent(
            picker, pln_ctx, drop, drop_pos, rotmat, cap.PLACE_DESCEND,
            'placing')
        placed_pos = drop_pos - np.array([0.0, 0.0, cap.PLACE_DESCEND],
                                          dtype=np.float32)
        release = None if place is None else solve_descent(
            picker, pln_ctx, place[-1], placed_pos, rotmat,
            -cap.RELEASE_RISE, 'rising after the release')
        if release is None:
            return False
        orientations[name] = {'drop': drop, 'place': place,
                              'release': release}
    if len(orientations) < len(cap.PLACINGS):
        print('not every placing could be solved')
        return False

    # the fixed poses: the same for every tube, chained so each IK is chosen
    # near the pose the arm comes from
    near = (orientations['towards']['drop'] if 'towards' in orientations
            else next(iter(orientations.values()))['drop'])
    print(f'fixed: push at {np.round(cap.PUSH_POS * 1000, 2)} mm, pop at '
          f'{np.round(cap.POP_POS * 1000, 2)} mm, re-pick at '
          f'{np.round(cap.REPICK_POS * 1000, 2)} mm, discard at '
          f'{np.round(cap.DISCARD_POS * 1000, 2)} mm')
    push_top = solve_pose(picker, pln_ctx, cap.PUSH_POS, cap.PUSH_ROTMAT,
                          near, 'push start')
    push = None if push_top is None else solve_descent(
        picker, pln_ctx, push_top, cap.PUSH_POS, cap.PUSH_ROTMAT,
        cap.PUSH_DESCEND, 'pushing')
    pop = None if push is None else solve_pose(
        picker, pln_ctx, cap.POP_POS, cap.POP_ROTMAT, push_top,
        'pop position')
    press = None if pop is None else solve_descent(
        picker, pln_ctx, pop, cap.POP_POS, cap.POP_ROTMAT, cap.POP_DIST,
        'pressing the lever')
    repick_top = None if press is None else solve_pose(
        picker, pln_ctx, cap.REPICK_POS, cap.REPICK_ROTMAT, near,
        're-pick point')
    repick = None if repick_top is None else solve_descent(
        picker, pln_ctx, repick_top, cap.REPICK_POS, cap.REPICK_ROTMAT,
        cap.REPICK_DESCEND, 're-picking')
    discard = None if repick is None else solve_pose(
        picker, pln_ctx, cap.DISCARD_POS, cap.DISCARD_ROTMAT, repick_top,
        'discard point')
    if discard is None:
        return False
    fixed = {'push_top': push_top, 'push': push, 'pop': pop, 'press': press,
             'repick_top': repick_top, 'repick': repick, 'discard': discard}

    deg = lambda qs: np.round(np.degrees(qs), 2)
    for name, entry in orientations.items():
        print(f'  {name:7s} drop    {deg(entry["drop"])}')
        print(f'  {"":7s} placed  {deg(entry["place"][-1])}  '
              f'({len(entry["place"]) - 1} samples)')
        print(f'  {"":7s} released{deg(entry["release"][-1])}  '
              f'({len(entry["release"]) - 1} samples)')
    for key, path_key, label in (('push_top', 'push', 'pushed'),
                                 ('pop', 'press', 'lever pressed'),
                                 ('repick_top', 'repick', 're-picked')):
        print(f'  {key:15s} {deg(fixed[key])}')
        print(f'  {label:15s} {deg(fixed[path_key][-1])}  '
              f'({len(fixed[path_key]) - 1} samples)')
    print(f'  {"discard":15s} {deg(discard)}')
    cpio.save(orientations, fixed, cap.inputs(), path=path)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=cpio.CAP_POSES_JSON,
                        help='where to write the JSON')
    args = parser.parse_args(argv)
    return 0 if precompute(path=args.out) else 1


if __name__ == '__main__':
    sys.exit(main())
