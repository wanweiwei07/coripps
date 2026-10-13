"""Turn TCP positions into a smooth joint sequence, or say why they cannot.

Given a robot, a point, and the orientation(s) worth trying, this solves IK at
every sample of the approach PATH -- the key poses joined by straight TCP
lines -- and picks the branch whose consecutive solutions are closest
together. A branch flip anywhere along the path would be a real move on the
real arm, so a path with one is rejected outright rather than executed.

Deliberately knows nothing about the tube grid: the caller supplies the
``path_fn`` (see ``layout_pick.pose_path``), so the same solver serves a place
or a cap-opener approach when those arrive.
"""

import numpy as np


# Two consecutive path samples are a few mm apart; more than this on any joint
# means IK jumped to a different branch, not that the arm travels.
MAX_STEP_JOINT_CHANGE = np.deg2rad(10.0)


def _joint_distance(qs_a, qs_b):
    diff = np.asarray(qs_a, dtype=np.float32) - np.asarray(qs_b, dtype=np.float32)
    diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(diff)


def _nearest_qs(qs_list, ref_qs):
    if qs_list is None:
        return None
    best_qs = None
    best_dist = np.inf
    for qs in qs_list:
        dist = float(np.linalg.norm(_joint_distance(qs, ref_qs)))
        if dist < best_dist:
            best_dist = dist
            best_qs = qs
    return best_qs


def tcp_rotmat_candidates(tcp_rotmat, point):
    """Normalize the ``tcp_rotmat`` argument into a tuple of orientations to
    try in order: either one fixed matrix, or a per-point callable returning a
    single matrix or a sequence of them."""
    if not callable(tcp_rotmat):
        return (tcp_rotmat,)
    resolved = tcp_rotmat(point)
    return tuple(resolved) if isinstance(resolved, (tuple, list)) else (resolved,)


def _grasp_tcp(robot, tcp):
    """The TCP to solve for; a tube picker's gripper grasp centre by default."""
    return robot.gripper.tcp('grasp_center') if tcp is None else tcp


def _segment_of(path_idx, marks):
    """Which key pose a failing path index belongs to: the segment ARRIVING
    at that key pose (index 0 is the first key pose itself)."""
    return int(np.searchsorted(marks, path_idx))


def diagnose(robot, tcp_rotmat, point, path_fn,
             max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """Solve one path; return ``(path, marks, diagnosis)``.

    ``path_fn(point, tcp_rotmat)`` gives ``(positions, marks)`` -- see
    ``layout_pick.pose_path``. ``diagnosis['reason']`` is one of ``ok`` /
    ``no_ik`` / ``ik_error`` / ``joint_jump``; ``diagnosis['pose_idx']`` is
    the key pose whose segment failed -- that is what makes a failing grid
    legible instead of just smaller.
    """
    positions, marks = path_fn(point, tcp_rotmat)
    qs_options = []
    for path_idx, position in enumerate(positions):
        try:
            qs_list = robot.ik(tgt_pos=position, tgt_rotmat=tcp_rotmat,
                               tcp=_grasp_tcp(robot, tcp), max_solutions=8)
        except IndexError:
            return None, marks, {'reason': 'ik_error',
                                 'pose_idx': _segment_of(path_idx, marks)}
        if qs_list is None or len(qs_list) == 0:
            return None, marks, {'reason': 'no_ik',
                                 'pose_idx': _segment_of(path_idx, marks)}
        qs_options.append(qs_list)

    best_path = None
    best_score = np.inf
    jump_counts = np.zeros(len(marks), dtype=np.int32)
    for first_qs in qs_options[0]:
        path = [first_qs]
        score = 0.0
        ref_qs = first_qs
        valid = True
        for path_idx, qs_list in enumerate(qs_options[1:], start=1):
            qs = _nearest_qs(qs_list, ref_qs)
            delta = _joint_distance(qs, ref_qs)
            if np.any(delta > max_joint_change):
                jump_counts[_segment_of(path_idx, marks)] += 1
                valid = False
                break
            score += float(np.linalg.norm(delta))
            path.append(qs)
            ref_qs = qs
        if valid and score < best_score:
            best_score = score
            best_path = path
    if best_path is None:
        return None, marks, {'reason': 'joint_jump',
                             'pose_idx': int(np.argmax(jump_counts))}
    return (np.asarray(best_path, dtype=np.float32), marks,
            {'reason': 'ok', 'pose_idx': None})


def follow_positions(robot, start_qs, positions, tcp_rotmat,
                     max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """Chain IK through ``positions`` from a GIVEN start config.

    Unlike :func:`diagnose`, the branch is not chosen here: it is whatever
    ``start_qs`` is on, and every next sample takes the IK solution nearest
    the previous one. Returns the joint path (``start_qs`` first, then one
    config per position) or None when a sample has no IK or would jump more
    than ``max_joint_change`` on any joint.
    """
    path = [np.asarray(start_qs, dtype=np.float32)]
    for position in positions:
        qs_list = robot.ik(tgt_pos=position, tgt_rotmat=tcp_rotmat,
                           tcp=_grasp_tcp(robot, tcp), max_solutions=8)
        if qs_list is None or len(qs_list) == 0:
            return None
        qs = _nearest_qs(qs_list, path[-1])
        if np.any(_joint_distance(qs, path[-1]) > max_joint_change):
            return None
        path.append(np.asarray(qs, dtype=np.float32))
    return np.asarray(path, dtype=np.float32)


def solve(robot, tcp_rotmat, point, path_fn,
          max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """``(path, marks)`` for one point and one orientation; path None on
    failure."""
    path, marks, _ = diagnose(robot, tcp_rotmat, point, path_fn,
                              max_joint_change=max_joint_change, tcp=tcp)
    return path, marks


def filter_valid(robot, points, tcp_rotmat, path_fn,
                 max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """Split ``points`` into solvable / unsolvable ones.

    Each point is tried against every orientation ``tcp_rotmat`` offers for it
    and the first that solves wins; the point is invalid only when none does.
    Returns the valid points, their joint PATHS ``(n, n_samples, ndof)``, the
    key-pose ``marks`` into those paths, the ORIENTATION each one was solved
    with (replay needs it -- the grid mixes orientations), and the invalid
    points.
    """
    valid_points, valid_paths, valid_rotmats, invalid_points = [], [], [], []
    marks = None
    for point in points:
        for rotmat in tcp_rotmat_candidates(tcp_rotmat, point):
            path, point_marks = solve(robot, rotmat, point, path_fn,
                                      max_joint_change=max_joint_change,
                                      tcp=tcp)
            if marks is None:
                marks = point_marks
            elif not np.array_equal(marks, point_marks):
                raise ValueError('path_fn returned different key-pose marks '
                                 'for different points -- paths must share '
                                 'one layout')
            if path is not None:
                valid_points.append(point)
                valid_paths.append(path)
                valid_rotmats.append(rotmat)
                break
        else:
            invalid_points.append(point)
    n_samples = 0 if marks is None else int(marks[-1]) + 1
    valid_points = np.asarray(valid_points, dtype=np.float32).reshape((-1, 3))
    invalid_points = np.asarray(invalid_points, dtype=np.float32).reshape((-1, 3))
    valid_rotmats = np.asarray(valid_rotmats, dtype=np.float32).reshape((-1, 3, 3))
    valid_paths = np.asarray(valid_paths, dtype=np.float32).reshape(
        (-1, n_samples, robot.ndof))
    return valid_points, valid_paths, marks, valid_rotmats, invalid_points


def print_diagnostics(robot, points, tcp_rotmat, path_fn, pose_names,
                      max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """Count why the grid fails, per key-pose segment and per x column."""
    no_ik = np.zeros(len(pose_names), dtype=np.int32)
    ik_error = np.zeros(len(pose_names), dtype=np.int32)
    jumps = np.zeros(len(pose_names), dtype=np.int32)
    reason_counts = {'ok': 0, 'no_ik': 0, 'ik_error': 0, 'joint_jump': 0}
    x_stats = {}

    for point in points:
        # try the orientations in preference order; a point fails only when all
        # of them do, and the reported diagnosis is the LAST (fallback) attempt
        for rotmat in tcp_rotmat_candidates(tcp_rotmat, point):
            path, _marks, diag = diagnose(robot, rotmat, point, path_fn,
                                          max_joint_change=max_joint_change,
                                          tcp=tcp)
            if path is not None:
                break
        reason = diag['reason']
        reason_counts[reason] += 1
        stats = x_stats.setdefault(float(point[0]), {
            'total': 0, 'ok': 0, 'no_ik': 0, 'ik_error': 0, 'joint_jump': 0})
        stats['total'] += 1
        stats[reason] += 1
        if path is not None:
            continue
        counts = {'no_ik': no_ik, 'ik_error': ik_error, 'joint_jump': jumps}
        counts[reason][diag['pose_idx']] += 1

    print('pick diagnostics:')
    print(f"  total={len(points)} ok={reason_counts['ok']} "
          f"no_ik={reason_counts['no_ik']} "
          f"ik_error={reason_counts['ik_error']} "
          f"joint_jump={reason_counts['joint_jump']}")
    for idx, name in enumerate(pose_names):
        print(f'  -> {name}: no_ik={int(no_ik[idx])} '
              f'ik_error={int(ik_error[idx])} joint_jump={int(jumps[idx])}')
    print('  by x:')
    for x_key in sorted(x_stats):
        stats = x_stats[x_key]
        print(f"    x={x_key: .4f}: total={stats['total']} ok={stats['ok']} "
              f"no_ik={stats['no_ik']} ik_error={stats['ik_error']} "
              f"joint_jump={stats['joint_jump']}")
    return reason_counts
