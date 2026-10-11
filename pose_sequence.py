"""Turn TCP positions into a smooth joint sequence, or say why they cannot.

Given a robot, a point, and the orientation(s) worth trying, this solves IK at
every pose of the approach and picks the branch whose consecutive solutions are
closest together -- an elbow flip between two poses of the same approach would
be a real move on the real arm, so a sequence with one is rejected outright
rather than executed.

Deliberately knows nothing about the tube grid: the caller supplies the
``pose_points_fn`` (see ``pick_layout.pose_points``), so the same solver serves
a place or a cap-opener approach when those arrive.
"""

import numpy as np


# Two consecutive poses of one approach are a short local move; more than this
# on any joint means IK jumped to a different branch, not that the arm travels.
MAX_STEP_JOINT_CHANGE = np.pi / 2.0


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


def diagnose(robot, tcp_rotmat, point, pose_points_fn,
             max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """Solve one sequence; return ``(sequence, diagnosis)``.

    ``diagnosis['reason']`` is one of ``ok`` / ``no_ik`` / ``ik_error`` /
    ``joint_jump``, with the pose index it failed at -- that is what makes a
    failing grid legible instead of just smaller.
    """
    points = pose_points_fn(point, tcp_rotmat)
    qs_options = []
    for pose_idx, pose_point in enumerate(points):
        try:
            qs_list = robot.ik(tgt_pos=pose_point, tgt_rotmat=tcp_rotmat,
                               tcp=_grasp_tcp(robot, tcp), max_solutions=8)
        except IndexError:
            return None, {'reason': 'ik_error', 'pose_idx': pose_idx,
                          'jump_pose_idx': None}
        if qs_list is None or len(qs_list) == 0:
            return None, {'reason': 'no_ik', 'pose_idx': pose_idx,
                          'jump_pose_idx': None}
        qs_options.append(qs_list)

    best_sequence = None
    best_score = np.inf
    jump_counts = np.zeros(len(points), dtype=np.int32)
    for first_qs in qs_options[0]:
        sequence = [first_qs]
        score = 0.0
        ref_qs = first_qs
        valid = True
        for pose_idx, qs_list in enumerate(qs_options[1:], start=1):
            qs = _nearest_qs(qs_list, ref_qs)
            delta = _joint_distance(qs, ref_qs)
            if np.any(delta > max_joint_change):
                jump_counts[pose_idx] += 1
                valid = False
                break
            score += float(np.linalg.norm(delta))
            sequence.append(qs)
            ref_qs = qs
        if valid and score < best_score:
            best_score = score
            best_sequence = sequence
    if best_sequence is None:
        jump_pose_idx = (int(np.argmax(jump_counts)) if np.any(jump_counts > 0)
                         else None)
        return None, {'reason': 'joint_jump', 'pose_idx': None,
                      'jump_pose_idx': jump_pose_idx}
    return np.asarray(best_sequence, dtype=np.float32), {
        'reason': 'ok', 'pose_idx': None, 'jump_pose_idx': None}


def solve(robot, tcp_rotmat, point, pose_points_fn,
          max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """The sequence for one point and one orientation, or None."""
    sequence, _ = diagnose(robot, tcp_rotmat, point, pose_points_fn,
                           max_joint_change=max_joint_change, tcp=tcp)
    return sequence


def filter_valid(robot, points, tcp_rotmat, pose_points_fn,
                 max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """Split ``points`` into solvable / unsolvable ones.

    Each point is tried against every orientation ``tcp_rotmat`` offers for it
    and the first that solves wins; the point is invalid only when none does.
    Returns the valid points, their joint sequences, the ORIENTATION each one
    was solved with (replay needs it -- the grid mixes orientations), and the
    invalid points.
    """
    valid_points, valid_sequences, valid_rotmats, invalid_points = [], [], [], []
    n_poses = 0
    for point in points:
        for rotmat in tcp_rotmat_candidates(tcp_rotmat, point):
            n_poses = max(n_poses, len(pose_points_fn(point, rotmat)))
            sequence = solve(robot, rotmat, point, pose_points_fn,
                             max_joint_change=max_joint_change, tcp=tcp)
            if sequence is not None:
                valid_points.append(point)
                valid_sequences.append(sequence)
                valid_rotmats.append(rotmat)
                break
        else:
            invalid_points.append(point)
    valid_points = np.asarray(valid_points, dtype=np.float32).reshape((-1, 3))
    invalid_points = np.asarray(invalid_points, dtype=np.float32).reshape((-1, 3))
    valid_rotmats = np.asarray(valid_rotmats, dtype=np.float32).reshape((-1, 3, 3))
    valid_sequences = np.asarray(valid_sequences, dtype=np.float32)
    if valid_sequences.size == 0:
        valid_sequences = valid_sequences.reshape((0, n_poses, robot.ndof))
    return valid_points, valid_sequences, valid_rotmats, invalid_points


def print_diagnostics(robot, points, tcp_rotmat, pose_points_fn, pose_names,
                      max_joint_change=MAX_STEP_JOINT_CHANGE, tcp=None):
    """Count why the grid fails, per pose and per x column."""
    no_ik = np.zeros(len(pose_names), dtype=np.int32)
    ik_error = np.zeros(len(pose_names), dtype=np.int32)
    jumps = np.zeros(len(pose_names), dtype=np.int32)
    reason_counts = {'ok': 0, 'no_ik': 0, 'ik_error': 0, 'joint_jump': 0}
    x_stats = {}

    for point in points:
        # try the orientations in preference order; a point fails only when all
        # of them do, and the reported diagnosis is the LAST (fallback) attempt
        for rotmat in tcp_rotmat_candidates(tcp_rotmat, point):
            sequence, diag = diagnose(robot, rotmat, point, pose_points_fn,
                                      max_joint_change=max_joint_change,
                                      tcp=tcp)
            if sequence is not None:
                break
        reason = diag['reason']
        reason_counts[reason] += 1
        stats = x_stats.setdefault(float(point[0]), {
            'total': 0, 'ok': 0, 'no_ik': 0, 'ik_error': 0, 'joint_jump': 0})
        stats['total'] += 1
        stats[reason] += 1
        if sequence is not None:
            continue
        if reason == 'no_ik':
            no_ik[diag['pose_idx']] += 1
        elif reason == 'ik_error':
            ik_error[diag['pose_idx']] += 1
        elif reason == 'joint_jump' and diag['jump_pose_idx'] is not None:
            jumps[diag['jump_pose_idx']] += 1

    print('pick diagnostics:')
    print(f"  total={len(points)} ok={reason_counts['ok']} "
          f"no_ik={reason_counts['no_ik']} "
          f"ik_error={reason_counts['ik_error']} "
          f"joint_jump={reason_counts['joint_jump']}")
    for idx, name in enumerate(pose_names):
        print(f'  pose {idx + 1} {name}: no_ik={int(no_ik[idx])} '
              f'ik_error={int(ik_error[idx])} joint_jump={int(jumps[idx])}')
    print('  by x:')
    for x_key in sorted(x_stats):
        stats = x_stats[x_key]
        print(f"    x={x_key: .4f}: total={stats['total']} ok={stats['ok']} "
              f"no_ik={stats['no_ik']} ik_error={stats['ik_error']} "
              f"joint_jump={stats['joint_jump']}")
    return reason_counts
