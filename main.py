import os
import sys
import numpy as np
import pyglet.window.key as pkey

# Allow running as `python riken/main.py` from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc
import one.utils.math as oum
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from coripps.robots.mobisys import mobisys


_RIKEN_DIR = os.path.dirname(os.path.abspath(__file__))
_TUBE_PATH = os.path.join(_RIKEN_DIR, 'models', 'tube.stl')
_POT_PATH = os.path.join(_RIKEN_DIR, 'models', 'pot.stl')
_TUBE_SCALE = (1e-3, 1e-3, 1e-3)
_POT_SCALE = None
_TUBE_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 2.0)
_POT_ROTMAT = np.eye(3, dtype=np.float32)
_PLACE_TUBE_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 2.0)
_TCP_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi)
_PLACE_TCP_ROTMAT = _TCP_ROTMAT
_POT_OBSERVE_TCP_ROTMAT = _PLACE_TCP_ROTMAT @ oum.rotmat_from_axangle(
    ouc.StandardAxis.Z, -np.pi / 2.0)
_LEAF_ROBOT_ROTMAT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, -np.pi / 2.0)
_LEAF_ROBOT_POS = np.array([0.42, 0.025, 0.0], dtype=np.float32)
_TUBE_POS_OFFSET = np.array([0.0, 0.0, -0.05], dtype=np.float32)
_CAP_OPENER_POINT = np.array([0.25761917, -0.04625, 0.09], dtype=np.float32)
_MM = 1e-3
_POT_XS_MM = (380, 480, 580, 680, 780)
_POT_Y_MM = -120
_POT_Z_MM = -43
_POT_OBSERVE_Z_OFFSET_MM = 170
_POT_SOLUTION_X_MM = 580
_PICK_X_MIN = -0.21
_PICK_X_MAX = 0.30
_PICK_X_STEP = 0.03
_PICK_Y_MIN = -0.06
_PICK_Y_MAX = 0.24
_PICK_Y_STEP = 0.03
_APPROACH_OFFSET_X = np.array([-0.03, 0.0, 0.0], dtype=np.float32)
_APPROACH_OFFSET_Z = np.array([0.0, 0.0, 0.03], dtype=np.float32)
_LIFT_OFFSET_Z = np.array([0.0, 0.0, 0.06], dtype=np.float32)
_MAX_STEP_JOINT_CHANGE = np.pi / 2.0
_POSE_NAMES = ('pre_pick_high', 'pre_pick', 'pick_place', 'lift')


def _mm_pos(x, y, z):
    return np.array([x, y, z], dtype=np.float32) * _MM


def pot_positions():
    """Return the configured pot positions in meters."""
    return np.asarray([
        _mm_pos(x_mm, _POT_Y_MM, _POT_Z_MM)
        for x_mm in _POT_XS_MM
    ], dtype=np.float32)


def pot_observation_points():
    """Return each pot observation position, 70mm above the pot."""
    return np.asarray([
        _mm_pos(x_mm, _POT_Y_MM, _POT_Z_MM + _POT_OBSERVE_Z_OFFSET_MM)
        for x_mm in _POT_XS_MM
    ], dtype=np.float32)


def pot_observation_point_by_x(x_mm):
    """Return the observation point for one configured pot x position."""
    if x_mm not in _POT_XS_MM:
        raise ValueError(f'x_mm must be one of {_POT_XS_MM}, got {x_mm}')
    return _mm_pos(x_mm, _POT_Y_MM, _POT_Z_MM + _POT_OBSERVE_Z_OFFSET_MM)


def tube_picker_pick_points_xy(z=0.10):
    """Return TCP pick candidate positions on the configured XY grid."""
    x_values = np.arange(
        _PICK_X_MIN,
        _PICK_X_MAX + _PICK_X_STEP * 0.5,
        _PICK_X_STEP,
        dtype=np.float32,
    )
    y_values = np.arange(
        _PICK_Y_MIN,
        _PICK_Y_MAX + _PICK_Y_STEP * 0.5,
        _PICK_Y_STEP,
        dtype=np.float32,
    )
    points = [
        (float(x), float(y), float(z))
        for y in y_values
        for x in x_values
        if not (x < 0.0 and y < 0.0)
    ]
    return np.asarray(points, dtype=np.float32)


def tube_picker_cap_opener_point():
    """Return the fixed TCP cap opener position."""
    return _CAP_OPENER_POINT.copy()


def tube_picker_pose_points(point):
    """Return four TCP positions used to validate and display one pick point."""
    return np.asarray([
        point + _APPROACH_OFFSET_X + _APPROACH_OFFSET_Z,
        point + _APPROACH_OFFSET_X,
        point,
        point + _LIFT_OFFSET_Z,
    ], dtype=np.float32)


def tube_picker_place_pose_points(point):
    """Return two TCP positions used to validate and display the place point."""
    return np.asarray([
        point + _LIFT_OFFSET_Z,
        point,
    ], dtype=np.float32)


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


def diagnose_pose_sequence(robot, tcp_rotmat, point, pose_points_fn=tube_picker_pose_points,
                           max_joint_change=_MAX_STEP_JOINT_CHANGE):
    """Solve one sequence and return a small failure diagnosis."""
    pose_points = pose_points_fn(point)
    qs_options = []
    for pose_idx, pose_point in enumerate(pose_points):
        try:
            qs_list = robot.ik_tcp(
                tgt_rotmat=tcp_rotmat,
                tgt_pos=pose_point,
                max_solutions=8,
            )
        except IndexError:
            return None, {
                'reason': 'ik_error',
                'pose_idx': pose_idx,
                'jump_pose_idx': None,
            }
        if qs_list is None or len(qs_list) == 0:
            return None, {
                'reason': 'no_ik',
                'pose_idx': pose_idx,
                'jump_pose_idx': None,
            }
        qs_options.append(qs_list)

    best_sequence = None
    best_score = np.inf
    jump_counts = np.zeros(len(pose_points), dtype=np.int32)
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
        jump_pose_idx = int(np.argmax(jump_counts)) if np.any(jump_counts > 0) else None
        return None, {
            'reason': 'joint_jump',
            'pose_idx': None,
            'jump_pose_idx': jump_pose_idx,
        }
    return np.asarray(best_sequence, dtype=np.float32), {
        'reason': 'ok',
        'pose_idx': None,
        'jump_pose_idx': None,
    }


def solve_pose_sequence(robot, tcp_rotmat, point, pose_points_fn=tube_picker_pose_points,
                        max_joint_change=_MAX_STEP_JOINT_CHANGE):
    """Solve the four-pose sequence for one point.

    The point is valid only when all four TCP positions have IK and adjacent
    solutions are close enough for a smooth local move.
    """
    sequence, _ = diagnose_pose_sequence(
        robot,
        tcp_rotmat,
        point,
        pose_points_fn=pose_points_fn,
        max_joint_change=max_joint_change,
    )
    return sequence


def print_pick_diagnostics(robot, pick_points, tcp_rotmat):
    no_ik_counts = np.zeros(len(_POSE_NAMES), dtype=np.int32)
    ik_error_counts = np.zeros(len(_POSE_NAMES), dtype=np.int32)
    jump_counts = np.zeros(len(_POSE_NAMES), dtype=np.int32)
    reason_counts = {'ok': 0, 'no_ik': 0, 'ik_error': 0, 'joint_jump': 0}
    x_stats = {}

    for point in pick_points:
        sequence, diag = diagnose_pose_sequence(robot, tcp_rotmat, point)
        reason = diag['reason']
        reason_counts[reason] += 1
        x_key = float(point[0])
        if x_key not in x_stats:
            x_stats[x_key] = {'total': 0, 'ok': 0, 'no_ik': 0, 'ik_error': 0, 'joint_jump': 0}
        x_stats[x_key]['total'] += 1
        x_stats[x_key][reason] += 1
        if sequence is not None:
            continue
        if reason == 'no_ik':
            no_ik_counts[diag['pose_idx']] += 1
        elif reason == 'ik_error':
            ik_error_counts[diag['pose_idx']] += 1
        elif reason == 'joint_jump' and diag['jump_pose_idx'] is not None:
            jump_counts[diag['jump_pose_idx']] += 1

    print('pick diagnostics:')
    print(
        f"  total={len(pick_points)} ok={reason_counts['ok']} "
        f"no_ik={reason_counts['no_ik']} ik_error={reason_counts['ik_error']} "
        f"joint_jump={reason_counts['joint_jump']}"
    )
    for pose_idx, pose_name in enumerate(_POSE_NAMES):
        print(
            f'  pose {pose_idx + 1} {pose_name}: '
            f'no_ik={int(no_ik_counts[pose_idx])} '
            f'ik_error={int(ik_error_counts[pose_idx])} '
            f'joint_jump={int(jump_counts[pose_idx])}'
        )
    print('  by x:')
    for x_key in sorted(x_stats):
        stats = x_stats[x_key]
        print(
            f"    x={x_key: .4f}: total={stats['total']} ok={stats['ok']} "
            f"no_ik={stats['no_ik']} ik_error={stats['ik_error']} "
            f"joint_jump={stats['joint_jump']}"
        )


def filter_valid_point_sequences(robot, points, tcp_rotmat):
    valid_points = []
    valid_sequences = []
    invalid_points = []
    for point in points:
        sequence = solve_pose_sequence(robot, tcp_rotmat, point)
        if sequence is None:
            invalid_points.append(point)
        else:
            valid_points.append(point)
            valid_sequences.append(sequence)
    valid_points = np.asarray(valid_points, dtype=np.float32).reshape((-1, 3))
    invalid_points = np.asarray(invalid_points, dtype=np.float32).reshape((-1, 3))
    valid_sequences = np.asarray(valid_sequences, dtype=np.float32)
    if valid_sequences.size == 0:
        valid_sequences = valid_sequences.reshape((0, len(_POSE_NAMES), 6))
    return valid_points, valid_sequences, invalid_points


def solve_cap_opener_sequence(robot, cap_opener_point, tcp_rotmat):
    return solve_pose_sequence(
        robot,
        tcp_rotmat,
        cap_opener_point,
        pose_points_fn=tube_picker_place_pose_points,
    )


def show_leaf_sampler_observations(scene, leaf_sampler, points):
    """Draw reachable/unreachable D405 observation points for leaf sampler."""
    results = []
    for idx, point in enumerate(points, start=1):
        qs, info, ok = leaf_sampler.ik_attached_frame(
            leaf_sampler.d405_frame,
            target_pos=point,
            axis_constraints={'z': np.array([0.0, 0.0, -1.0], dtype=np.float32)},
            selik_seed_count=128,
        )
        rgb = ouc.BasicColor.GREEN if ok else ouc.BasicColor.RED
        ossop.sphere(
            pos=point,
            radius=0.007,
            segments=12,
            rgb=rgb,
            alpha=1.0,
        ).attach_to(scene)
        if ok:
            sol_robot = leaf_sampler.clone()
            sol_robot.fk(qs=qs)
            sol_robot.alpha = 0.5
            sol_robot.attach_to(scene)
            sol_robot.toggle_attached_frames(
                'd405',
                length_scale=0.25,
                radius_scale=0.4,
                color_mat=ouc.CoordColor.DYO,
            )
        print(
            f'leaf_sampler pot_obs {idx}/{len(points)} '
            f'point={np.round(point, 4)} ok={ok} info={info}'
        )
        results.append((point, qs, info, ok))
    return results


def show_tube_picker_points():
    base = ovw.World(cam_pos=(1.25, 0.75, 0.65), cam_lookat_pos=(0.45, -0.10, 0.12))
    ossop.frame().attach_to(base.scene)

    mobisys_base, tube_handler, leaf_sampler = mobisys()
    mobisys_base.attach_to(base.scene)
    tube_handler.attach_to(base.scene)
    tube_handler.gripper.attach_to(base.scene)
    tube_handler.toggle_tcp(color_mat=ouc.CoordColor.MYC)
    leaf_sampler.attach_to(base.scene)
    leaf_sampler.toggle_attached_frames(
        'd405',
        length_scale=0.3,
        radius_scale=0.5,
        color_mat=ouc.CoordColor.MYC,
    )

    pot_points = pot_positions()
    pot_obs_points = pot_observation_points()
    leaf_obs_results = show_leaf_sampler_observations(
        base.scene, leaf_sampler, pot_obs_points)
    tube_points = tube_picker_pick_points_xy(z=0.07)
    print('tube handler diagnostics:')
    print_pick_diagnostics(tube_handler, tube_points, _TCP_ROTMAT)
    tube_valid_points, tube_valid_sequences, tube_invalid_points = filter_valid_point_sequences(
        tube_handler, tube_points, _TCP_ROTMAT)
    tube_valid_mask = np.asarray([
        np.any(np.all(np.isclose(tube_valid_points, point), axis=1))
        for point in tube_points
    ], dtype=bool)
    tube_invalid_mask = np.asarray([
        np.any(np.all(np.isclose(tube_invalid_points, point), axis=1))
        for point in tube_points
    ], dtype=bool)
    tube_template = osso.SceneObject.from_file(
        _TUBE_PATH,
        scale=_TUBE_SCALE,
        collision_type=None,
        rgb=ouc.ExtendedColor.SILVER,
        alpha=0.5,
    )
    pot_template = osso.SceneObject.from_file(
        _POT_PATH,
        scale=_POT_SCALE,
        collision_type=None,
        rgb=ouc.ExtendedColor.CHINESE_RED,
        alpha=0.85,
    )

    for point in tube_points:
        tube = tube_template.clone()
        tube.set_rotmat_pos(rotmat=_TUBE_ROTMAT, pos=point + _TUBE_POS_OFFSET)
        tube.attach_to(base.scene)

    for point in tube_points[tube_invalid_mask]:
        ossop.sphere(
            pos=point,
            radius=0.0035,
            segments=8,
            rgb=ouc.BasicColor.RED,
            alpha=0.95,
        ).attach_to(base.scene)

    for point in tube_points[tube_valid_mask]:
        ossop.sphere(
            pos=point,
            radius=0.0035,
            segments=8,
            rgb=ouc.BasicColor.GREEN,
            alpha=0.95,
        ).attach_to(base.scene)

    for pot_point in pot_points:
        pot = pot_template.clone()
        pot.set_rotmat_pos(rotmat=_POT_ROTMAT, pos=pot_point)
        pot.attach_to(base.scene)

    cursor = ossop.sphere(
        pos=tube_valid_points[0] if len(tube_valid_points) > 0 else (0.0, 0.0, 0.0),
        radius=0.009,
        segments=12,
        rgb=ouc.BasicColor.YELLOW,
        alpha=1.0,
    )
    cursor.attach_to(base.scene)

    state = {'point_idx': 0, 'pose_idx': 0}

    def _show_current():
        if len(tube_valid_points) == 0:
            return
        point_idx = state['point_idx']
        pose_idx = state['pose_idx']
        pose_points = tube_picker_pose_points(tube_valid_points[point_idx])
        tube_handler.fk(qs=tube_valid_sequences[point_idx, pose_idx])
        cursor.set_rotmat_pos(pos=pose_points[pose_idx])
        print(
            f'tube point {point_idx + 1}/{len(tube_valid_points)} '
            f'pose {pose_idx + 1}/4 tcp={pose_points[pose_idx]}'
        )

    def _step(dt):
        if not base.input_manager.is_key_pressed_edge(pkey.SPACE):
            return
        if len(tube_valid_points) == 0:
            return
        state['pose_idx'] += 1
        if state['pose_idx'] >= 4:
            state['pose_idx'] = 0
            state['point_idx'] = (state['point_idx'] + 1) % len(tube_valid_points)
        _show_current()

    _show_current()
    base.schedule_interval(_step, interval=0.03)

    print(
        f'tube pick points: total={len(tube_points)} '
        f'tube_pick_valid={int(np.count_nonzero(tube_valid_mask))} '
        f'tube_pick_invalid={int(np.count_nonzero(tube_invalid_mask))} '
        f'pot_count={len(pot_points)} '
        f'max_joint_change={np.rad2deg(_MAX_STEP_JOINT_CHANGE):.1f}deg'
    )
    return (
        base,
        mobisys_base,
        tube_handler,
        leaf_sampler,
        pot_points,
        pot_obs_points,
        leaf_obs_results,
        tube_points,
        tube_valid_points,
        tube_valid_sequences,
        tube_invalid_points,
    )


if __name__ == '__main__':
    import builtins

    (
        base,
        mobisys_base,
        tube_handler,
        leaf_sampler,
        pot_points,
        pot_obs_points,
        leaf_obs_results,
        tube_points,
        tube_valid_points,
        tube_valid_sequences,
        tube_invalid_points,
    ) = show_tube_picker_points()
    builtins.base = base
    builtins.mobisys_base = mobisys_base
    builtins.tube_handler = tube_handler
    builtins.leaf_sampler = leaf_sampler
    builtins.pot_points = pot_points
    builtins.pot_obs_points = pot_obs_points
    builtins.leaf_obs_results = leaf_obs_results
    builtins.tube_points = tube_points
    builtins.tube_valid_points = tube_valid_points
    builtins.tube_valid_sequences = tube_valid_sequences
    builtins.tube_invalid_points = tube_invalid_points
    base.run()
